#include "pch.h"

#include "VirtualUpscalerDriver.h"

#include "DepthFinderCore.h"
#include "VirtualUpscalerMode.h"

#include <atomic>
#include <NVNGX_Parameter.h>
#include <State.h>
#include <Config.h>
#include <Util.h>
#include <upscalers/FeatureProvider_Dx12.h>
#include <upscalers/IFeature_Dx12.h>
#include <inputs/FG/Upscaler_Inputs_Dx12.h>
#include <dlssnr/DlssNr.h>
#include <dlssnr/DlssNrFeature_Dx12.h>

#include <d3dx/d3dx12.h>
#include <shaders/Shader_Common.h>
#include <shaders/Shader_Dx12.h>
#include <shaders/Shader_Dx12Utils.h>

// Sub-pixel shift of the picture before the backend reads it.
//
// DLSS/DLAA needs the frame it is given to be jittered: the jitter moves every pixel's sample position by a fraction of a
// pixel from frame to frame, and the model accumulates those sub-pixel samples into one clean edge. A game supplies that by
// offsetting its projection matrix. A game with no upscaler has no such matrix for us, so the shift is applied to the
// finished picture instead and the same offset is handed over as NVSDK_NGX_Parameter_Jitter_Offset_X/Y.
//
// The motion vectors stay unjittered (the producer estimates them on the unshifted picture), so the MVJittered create flag
// stays off.
constexpr char kJitterHlsl[] = R"(
cbuffer Params : register(b0)
{
    float OffX;
    float OffY;
    float InvW;
    float InvH;
};
Texture2D<float3> Source : register(t0);
RWTexture2D<float3> Dest : register(u0);

float3 Fetch(int2 c, int2 maxc)
{
    c = clamp(c, int2(0, 0), maxc);
    return Source.Load(int3(c, 0));
}

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    int2 size = int2(round(1.0f / InvW), round(1.0f / InvH));

    if (id.x >= (uint) size.x || id.y >= (uint) size.y)
        return;

    int2 maxc = size - 1;
    float2 p = float2(id.xy) - float2(OffX, OffY);
    float2 fl = floor(p);
    int2 i0 = int2(fl);
    float2 f = p - fl;

    float3 c0 = lerp(Fetch(i0, maxc), Fetch(i0 + int2(1, 0), maxc), f.x);
    float3 c1 = lerp(Fetch(i0 + int2(0, 1), maxc), Fetch(i0 + int2(1, 1), maxc), f.x);

    Dest[id.xy] = lerp(c0, c1, f.y);
}
)";

class JitterShift_Dx12 : public Shader_Dx12
{
  private:
    struct alignas(256) InternalConstants
    {
        float OffX;
        float OffY;
        float InvW;
        float InvH;
    };

    static constexpr int kJitterHeaps = 2;

    FrameDescriptorHeap _jitterHeaps[kJitterHeaps];
    int _jitterCounter = 0;

  public:
    JitterShift_Dx12(std::string name, ID3D12Device* device) : Shader_Dx12(name, device)
    {
        if (device == nullptr)
            return;

        if (!SetupRootSignature(device, 1, 1, 1))
        {
            LOG_WARN("Jitter shift: root signature failed");
            return;
        }

        ID3DBlob* blob = CompileShader(kJitterHlsl, "CSMain", "cs_5_0");

        if (blob == nullptr)
        {
            LOG_WARN("Jitter shift: the shader did not compile, jitter stays off");
            return;
        }

        D3D12_COMPUTE_PIPELINE_STATE_DESC desc {};
        desc.pRootSignature = _rootSignature;
        desc.CS = { blob->GetBufferPointer(), blob->GetBufferSize() };

        auto hr = device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&_pipelineState));
        blob->Release();

        if (hr != S_OK)
        {
            LOG_WARN("Jitter shift: pipeline state failed {0:x}", (unsigned int) hr);
            return;
        }

        D3D12_RESOURCE_DESC cbDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(InternalConstants));
        auto heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);

        hr = device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &cbDesc, D3D12_RESOURCE_STATE_GENERIC_READ,
                                             nullptr, IID_PPV_ARGS(&_constantBuffer));

        if (hr != S_OK)
        {
            LOG_WARN("Jitter shift: constants buffer failed {0:x}", (unsigned int) hr);
            return;
        }

        _init = InitHeaps(device, _jitterHeaps, kJitterHeaps);
    }

    bool Dispatch(ID3D12GraphicsCommandList* cmd, ID3D12Resource* src, ID3D12Resource* dst, float offX, float offY,
                  uint32_t width, uint32_t height)
    {
        if (!_init || cmd == nullptr || src == nullptr || dst == nullptr || width == 0 || height == 0)
            return false;

        _jitterCounter = (_jitterCounter + 1) % kJitterHeaps;
        FrameDescriptorHeap& heap = _jitterHeaps[_jitterCounter];

        CreateShaderResourceView(_device, src, heap.GetSrvCPU(0));
        CreateUnorderedAccessView(_device, dst, heap.GetUavCPU(0), 0);

        InternalConstants constants {};
        constants.OffX = offX;
        constants.OffY = offY;
        constants.InvW = 1.0f / (float) width;
        constants.InvH = 1.0f / (float) height;

        if (!CreateConstantsBuffer(_device, _constantBuffer, constants, heap.GetCbvCPU(0)))
            return false;

        ID3D12DescriptorHeap* heaps[] = { heap.GetHeapCSU() };
        cmd->SetDescriptorHeaps(_countof(heaps), heaps);
        cmd->SetComputeRootSignature(_rootSignature);
        cmd->SetPipelineState(_pipelineState);
        cmd->SetComputeRootDescriptorTable(0, heap.GetTableGPUStart());
        cmd->Dispatch((width + 15) / 16, (height + 15) / 16, 1);

        return true;
    }
};

// Halton(2,3): the low-discrepancy sequence DLSS's own jitter uses, in the -0.5..0.5 pixel range.
float HaltonSequence(int index, int base)
{
    float f = 1.0f;
    float r = 0.0f;

    for (int i = index; i > 0; i /= base)
    {
        f /= (float) base;
        r += f * (float) (i % base);
    }

    return r;
}

// One instance per device, for the driver's life.
// Disabled: shifting the finished picture is resampling, not jitter, so it cannot add the sub-pixel samples DLSS needs. The
// real jitter has to reach the game's projection matrix (hooks/D3D11_Hooks.cpp, CbProbe).
JitterShift_Dx12* JitterShiftFor(ID3D12Device* device)
{
    (void) device;
    return nullptr;
}

namespace native
{

namespace
{
// The feature the virtual upscaler is running now, for IsVirtualUpscalerFeature (read by the menu thread).
std::atomic<const void*> g_virtualFeature { nullptr };

// Our feature's handle id: never in the real NGX handle tables (HandleToFeature, Dx12Contexts), so no game call finds it. It is
// in State::changeBackend, so the menu's backend switch reaches it like any other feature.
constexpr UINT kVirtualHandleId = 0x4E524631; // 'NRF1'

// Releases a texture once the GPU can no longer be using it, by the same delay Util::DelayedDestroy gives features.
void ReleaseTexture(ID3D12Resource*& texture)
{
    Util::DelayedRelease(texture);
    texture = nullptr;
}

void Transition(ID3D12GraphicsCommandList* cmd, ID3D12Resource* resource, D3D12_RESOURCE_STATES before,
                D3D12_RESOURCE_STATES after)
{
    if (before == after)
        return;

    D3D12_RESOURCE_BARRIER b {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = resource;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = before;
    b.Transition.StateAfter = after;
    cmd->ResourceBarrier(1, &b);
}

// UAVs cannot be sRGB; the bytes are the same, so the copy in and out keeps the picture exact.
DXGI_FORMAT UavFormat(DXGI_FORMAT view)
{
    switch (view)
    {
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
        return DXGI_FORMAT_R8G8B8A8_UNORM;
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
        return DXGI_FORMAT_B8G8R8A8_UNORM;
    case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB:
        return DXGI_FORMAT_B8G8R8X8_UNORM;
    default:
        return view;
    }
}

bool IsUnrealGame()
{
    const auto& state = State::Instance();
    return state.NVNGX_Engine == NVSDK_NGX_ENGINE_TYPE_UNREAL || state.gameEngine == GameEngineType::Unreal ||
           state.gameQuirks & GameQuirk::ForceUnrealEngine;
}

// The FSR and XeSS backends take each input in the state its *ResourceBarrier setting names (and hand it back in it), and in
// an Unreal game set colour and motion to RENDER_TARGET and UNORDERED_ACCESS on their first Evaluate. Setting those here,
// before UpscaleStart, makes frame generation's copies (which read the same settings) agree with the backend from the first
// frame. DLSS takes everything in its working state.
bool UsesBarrierSettings(Upscaler backend) { return backend != Upscaler::DLSS && backend != Upscaler::DLSSD; }

D3D12_RESOURCE_STATES SettingOr(const CustomOptional<int32_t, NoDefault>& setting, D3D12_RESOURCE_STATES working)
{
    return setting.has_value() ? (D3D12_RESOURCE_STATES) setting.value() : working;
}

Upscaler WantedBackend()
{
    const auto& cfg = *Config::Instance();
    const Upscaler wanted = cfg.Dx12Upscaler.has_value() ? cfg.Dx12Upscaler.value() : Upscaler::FFX;

    // Ray Reconstruction needs inputs we do not have; the config stores it as DLSS anyway.
    return wanted == Upscaler::DLSSD ? Upscaler::DLSS : wanted;
}
} // namespace

VirtualUpscalerDriver::VirtualUpscalerDriver() = default;

bool IsVirtualUpscalerFeature(const void* feature) { return feature != nullptr && g_virtualFeature.load() == feature; }

VirtualUpscalerDriver::~VirtualUpscalerDriver()
{
    Release();
    ReleaseTexture(_output);
    ReleaseTexture(_input);
    ReleaseTexture(_jittered);
    ReleaseTexture(_neutralDepth);

    if (_rtvHeap != nullptr)
        _rtvHeap->Release();

    delete _params;
}

void VirtualUpscalerDriver::DropFeature(bool destroyFgContext)
{
    auto& state = State::Instance();

    // What NVSDK_NGX_D3D12_ReleaseFeature (destroyFgContext) and FeatureProvider_Dx12::ChangeFeature (not) do for a game's
    // feature.
    if (state.currentFG != nullptr && state.activeFgInput == FGInput::Upscaler)
    {
        state.fgChanged = true;
        state.clearCapturedHudlesses = true;

        if (destroyFgContext)
            state.currentFG->DestroyFGContext();
    }

    UpscalerInputsDx12::Reset();

    if (state.currentFeature == _feature.get())
        state.currentFeature = nullptr;

    if (g_virtualFeature.load() == _feature.get())
        g_virtualFeature = nullptr;

    Util::DelayedDestroy(std::move(_feature));
}

void VirtualUpscalerDriver::Release()
{
    if (_feature == nullptr)
        return;

    _paused = false;

    LOG_INFO("Virtual upscaler: releasing the {} backend", _backendName);
    DropFeature(true);
    State::Instance().changeBackend.erase(kVirtualHandleId);
    _backendName.clear();
}

void VirtualUpscalerDriver::Pause()
{
    if (_feature == nullptr || _paused)
        return;

    _paused = true;

    auto& state = State::Instance();

    if (state.currentFeature == _feature.get())
        state.currentFeature = nullptr;

    LOG_INFO("Virtual upscaler: paused, the game is calling its own upscaler (the {} backend stays)", _backendName);
}

bool VirtualUpscalerDriver::CreateFeature(ID3D12GraphicsCommandList* cmd, const Key& key, Upscaler backend)
{
    _params->Set(NVSDK_NGX_Parameter_Width, key.width);
    _params->Set(NVSDK_NGX_Parameter_Height, key.height);
    _params->Set(NVSDK_NGX_Parameter_OutWidth, key.width);
    _params->Set(NVSDK_NGX_Parameter_OutHeight, key.height);
    _params->Set(NVSDK_NGX_Parameter_PerfQualityValue, 1);

    // Not jittered and not low-res: the guides are output-sized, unjittered motion. AutoExposure because there is no exposure
    // texture to give: without both, FFXFeatureDx12::EvaluateInternal returns true having dispatched nothing.
    unsigned int flags = NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;

    if (key.depthReversed)
        flags |= NVSDK_NGX_DLSS_Feature_Flags_DepthInverted;

    if (key.hdr)
        flags |= NVSDK_NGX_DLSS_Feature_Flags_IsHDR;

    _params->Set(NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags, flags);

    // As TryCreateOptiFeature does: the menu and some backends read the API of the feature being run from here.
    State::Instance().api = API::DX12;

    std::unique_ptr<IFeature_Dx12> feature;

    // A backend library can throw (the FSR upscaler does on a size it cannot use): not into the game's present.
    try
    {
        if (!FeatureProvider_Dx12::GetFeature(backend, kVirtualHandleId, _params, &feature) || feature == nullptr)
        {
            _error = std::format("could not create the {} backend", UpscalerDisplayName(backend));
            LOG_WARN("Virtual upscaler: {}", _error);
            return false;
        }

        if (!feature->Init(_device, cmd, _params))
        {
            _error = std::format("the {} backend failed to initialise", feature->Name());
            LOG_WARN("Virtual upscaler: {}", _error);
            Util::DelayedDestroy(std::move(feature));
            return false;
        }
    }
    catch (...)
    {
        _error = std::format("the {} backend threw an exception while starting", UpscalerDisplayName(backend));
        LOG_ERROR("Virtual upscaler: {}", _error);

        if (feature != nullptr)
            Util::DelayedDestroy(std::move(feature));

        return false;
    }

    _feature = std::move(feature);
    g_virtualFeature = _feature.get();
    return true;
}

bool VirtualUpscalerDriver::EnsureFeature(ID3D12GraphicsCommandList* cmd, const Key& key, bool rebuild, bool realDepth)
{
    if (_feature != nullptr && !rebuild)
    {
        const bool flagsOnly = key.width == _featureKey.width && key.height == _featureKey.height &&
                               key.format == _featureKey.format && key.backend == _featureKey.backend;

        // Built before the game's depth came, the direction was a guess: the first real depth corrects it at once
        if (!_settle.Now(_featureKey, key, flagsOnly, !_featureRealDepth && realDepth))
        {
            // A guess that was right: from now on a flip is a real one, and waits
            if (realDepth && key == _featureKey)
                _featureRealDepth = true;

            return true;
        }
    }

    // Tried and failed for exactly this: not again every frame.
    if (_feature == nullptr && !rebuild && _failed && key == _failedKey)
        return false;

    const bool replacing = _feature != nullptr;

    if (replacing)
    {
        LOG_INFO("Virtual upscaler: rebuilding the {} backend", _backendName);
        DropFeature(false);
    }

    const auto backend = (Upscaler) key.backend;

    // Like the real path's fallback when a chosen backend will not initialise; FFX needs nothing from the game.
    if (!CreateFeature(cmd, key, backend) && (backend == Upscaler::FFX || !CreateFeature(cmd, key, Upscaler::FFX)))
    {
        _failed = true;
        _failedKey = key;
        return false;
    }

    _featureRealDepth = realDepth;
    _failed = false;
    auto& state = State::Instance();

    // GetFeature recorded what it built in Dx12Upscaler, which WantedBackend reads: the key matches it from the next frame on.
    _featureKey = key;
    _featureKey.backend = (int) WantedBackend();
    _backendName = _feature->Name();
    _error.clear();

    state.changeBackend[kVirtualHandleId] = false;

    if (replacing && state.currentFG != nullptr && state.activeFgInput == FGInput::Upscaler)
        state.currentFG->UpdateTarget();

    LOG_INFO("Virtual upscaler: {} ready, {}x{}, depth reversed {}, HDR {}", _backendName, key.width, key.height,
             key.depthReversed, key.hdr);
    return true;
}

bool VirtualUpscalerDriver::EnsureTexture(ID3D12Resource*& texture, Key& made, const Key& key,
                                          D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES restState, const char* what)
{
    if (texture != nullptr && made.width == key.width && made.height == key.height && made.format == key.format)
        return true;

    ReleaseTexture(texture);

    D3D12_HEAP_PROPERTIES heap {};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC desc {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = key.width;
    desc.Height = key.height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = key.format;
    desc.SampleDesc.Count = 1;
    desc.Flags = flags;

    const HRESULT hr = _device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, restState, nullptr,
                                                        IID_PPV_ARGS(&texture));

    if (FAILED(hr))
    {
        texture = nullptr;
        _error = std::format("creating the {}x{} {} texture (format {}) failed: {:X}", key.width, key.height, what,
                             (int) key.format, (UINT) hr);
        LOG_WARN("Virtual upscaler: {}", _error);
        return false;
    }

    made = key;
    return true;
}

ID3D12Resource* VirtualUpscalerDriver::NeutralDepth(ID3D12GraphicsCommandList* cmd, const Key& key)
{
    constexpr auto kRead = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

    Key wanted = key;
    wanted.format = DXGI_FORMAT_R32_FLOAT;
    ID3D12Resource* before = _neutralDepth;

    if (!EnsureTexture(_neutralDepth, _neutralKey, wanted, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                       D3D12_RESOURCE_STATE_RENDER_TARGET, "neutral depth"))
        return nullptr;

    if (_neutralDepth != before)
        _neutralFilled = false;

    if (_neutralFilled && _neutralReversed == key.depthReversed)
        return _neutralDepth;

    if (_rtvHeap == nullptr)
    {
        D3D12_DESCRIPTOR_HEAP_DESC heap {};
        heap.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        heap.NumDescriptors = 1;

        if (FAILED(_device->CreateDescriptorHeap(&heap, IID_PPV_ARGS(&_rtvHeap))))
        {
            _rtvHeap = nullptr;
            _error = "creating the neutral depth's descriptor heap failed";
            return nullptr;
        }
    }

    if (_neutralFilled)
        Transition(cmd, _neutralDepth, kRead, D3D12_RESOURCE_STATE_RENDER_TARGET);

    const auto rtv = _rtvHeap->GetCPUDescriptorHandleForHeapStart();
    _device->CreateRenderTargetView(_neutralDepth, nullptr, rtv);
    const float farDepth[4] = { key.depthReversed ? 0.0f : 1.0f, 0.0f, 0.0f, 0.0f };
    cmd->ClearRenderTargetView(rtv, farDepth, 0, nullptr);
    Transition(cmd, _neutralDepth, D3D12_RESOURCE_STATE_RENDER_TARGET, kRead);

    _neutralFilled = true;
    _neutralReversed = key.depthReversed;
    return _neutralDepth;
}

bool VirtualUpscalerDriver::Run(ID3D12GraphicsCommandList* cmd, const NativeFrame& frame, ID3D12CommandQueue* queue,
                                DXGI_COLOR_SPACE_TYPE colorSpace, uint64_t epoch)
{
    if (cmd == nullptr || frame.color == nullptr || frame.motion == nullptr)
        return false;

    if (_device == nullptr)
    {
        Microsoft::WRL::ComPtr<ID3D12Device> device;

        if (FAILED(cmd->GetDevice(IID_PPV_ARGS(&device))))
        {
            _error = "could not get the D3D12 device from the command list";
            return false;
        }

        _device = device.Get(); // not kept alive by us: lives as long as the game's device
    }

    if (_params == nullptr)
        _params = new NVNGX_Parameters(API::DX12, false);

    const D3D12_RESOURCE_DESC colorDesc = frame.color->GetDesc();

    Key key;
    key.width = (uint32_t) colorDesc.Width;
    key.height = colorDesc.Height;
    key.format = UavFormat(frame.colorFormat != DXGI_FORMAT_UNKNOWN ? frame.colorFormat : colorDesc.Format);
    // A frame with no depth keeps the convention the backend was built with, so it is not rebuilt for it.
    key.depthReversed = frame.depth != nullptr || _feature == nullptr ? frame.depthReversed : _featureKey.depthReversed;
    key.hdr = frame.space == ColorSpace::ScRgb; // PQ is a perceptual encoding, as sRGB is: passed as it comes
    key.backend = (int) WantedBackend();

    // No usable size yet (a swapchain still being made or resized; the FSR backend throws below a few pixels): wait for the next frame, build nothing.
    if (key.width < 16 || key.height < 16)
    {
        if (_zeroSizeFrames++ == 0)
            LOG_WARN("Virtual upscaler: the picture is {}x{}, too small, waiting for a real size", key.width, key.height);

        return false;
    }

    _zeroSizeFrames = 0;

    auto& state = State::Instance();
    auto& cfg = *Config::Instance();

    // Our own call, for all of it: GetFeature and Evaluate both tell the depth finder a game is calling an upscaler.
    SyntheticUpscalerCallScope guard;

    // A backend switch from the menu, or a backend asking to be rebuilt (FeatureProvider_Dx12::ChangeFeature's job for a game's
    // feature).
    bool rebuild = false;

    if (const auto change = state.changeBackend.find(kVirtualHandleId);
        _feature != nullptr && change != state.changeBackend.end() && change->second)
    {
        if (state.newBackend != Upscaler::Reset)
        {
            key.backend = (int) (state.newBackend == Upscaler::DLSSD ? Upscaler::DLSS : state.newBackend);
            state.newBackend = Upscaler::Reset;
        }

        change->second = false;
        rebuild = true;
    }

    if (!EnsureFeature(cmd, key, rebuild, frame.depth != nullptr))
        return false;

    const bool copyIn = colorDesc.Format != key.format;

    if (!EnsureTexture(_output, _outputKey, key, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "output"))
        return false;

    // UAV-capable: with the jitter on, this copy is written by JitterShift instead of CopyResource.
    if (copyIn &&
        !EnsureTexture(_input, _inputKey, key, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST,
                       "input"))
        return false;

    // The backend and frame generation need depth, NR does not: with none, they get an all-far depth and NR gets none.
    ID3D12Resource* depth = frame.depth != nullptr ? frame.depth : NeutralDepth(cmd, key);

    if (depth == nullptr)
        return false;

    // Frame generation's Upscaler input only has a device once NVSDK_NGX_D3D12_Init gave it one, which a game with no upscaler
    // never calls.
    UpscalerInputsDx12::Init(_device);

    const auto backend = _feature->GetUpscalerType();
    const bool usesSettings = UsesBarrierSettings(backend);

    // The picture is a render target in a D3D12 Unreal game; a D3D11 game's shared picture is not (no
    // ALLOW_RENDER_TARGET), and moving it to that state makes the list invalid.
    if (usesSettings && IsUnrealGame())
    {
        // Set either way: the FSR backends add the render-target state themselves for an Unreal game when it is unset.
        if (!cfg.ColorResourceBarrier.has_value())
            cfg.ColorResourceBarrier.set_volatile_value((colorDesc.Flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET) != 0
                                                            ? D3D12_RESOURCE_STATE_RENDER_TARGET
                                                            : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

        if (!cfg.MVResourceBarrier.has_value())
            cfg.MVResourceBarrier.set_volatile_value(D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }

    constexpr auto kRead = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    constexpr auto kWrite = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

    // Frame generation's copies (UpscaleStart/UpscaleEnd) read the settings whatever the backend.
    const auto fgMotion = SettingOr(cfg.MVResourceBarrier, kRead);
    const auto fgDepth = SettingOr(cfg.DepthResourceBarrier, kRead);
    const auto fgOutput = SettingOr(cfg.OutputResourceBarrier, kWrite);

    const auto inColor = usesSettings ? SettingOr(cfg.ColorResourceBarrier, kRead) : kRead;
    const auto inMotion = usesSettings ? fgMotion : kRead;
    const auto inDepth = usesSettings ? fgDepth : kRead;
    const auto inOutput = usesSettings ? fgOutput : kWrite;

    // What happens to the picture this frame (VirtualUpscalerMode.h): nothing (NR off, no pass asked for), NR on it in place
    // with the backend only counted, or the backend's pass with NR (when on) around it as for a game's call.
    const auto route = DecideVuNrRoute(
        cfg.DlssNrEnabled.value_or_default(),
        VuPassSettingFromIni(cfg.DlssNrNativeUpscalerPass.has_value(), cfg.DlssNrNativeUpscalerPass.value_or_default()));

    // NR on the picture itself comes first, before anything copies the picture: frame generation's HUD-less picture (the
    // copy into Output below, captured at UpscaleEnd) must be the one that is presented, NR included. Built from a picture
    // without NR, the generated frames would differ from the real ones in NR's changes, and the HUD fix would take those
    // changes for HUD. The picture is in the state it arrived in and the guides where the producer keeps them, as NR wants.
    bool nrRan = false;

    if (route == VuNrRoute::InPlace)
        nrRan = DlssNr::ApplyVirtualUpscalerInput(queue, cmd, frame.color, frame.depth, frame.motion,
                                                  frame.historyDistrust, frame.depthReversed, frame.reset, colorSpace,
                                                  frame.pictureState, epoch);
    else
        DlssNr::NativeInputSkipped(); // NR off or on the backend's output: in place, its history starts over next time

    // Only when NR really ran: the status line says so, and otherwise shows why it did not (FinishedPictureStatus).
    _nrInPlace.store(nrRan, std::memory_order_relaxed);

    // The colour the backend reads: the picture itself, or a typed copy of it. The picture stays in COPY_SOURCE until the
    // result is copied back over it.
    ID3D12Resource* color = frame.color;
    D3D12_RESOURCE_STATES pictureNow = frame.pictureState;

    // Sub-pixel jitter, only with the backend's own pass: a shifted picture handed straight on would make the whole image
    // wobble. Its own texture, because the picture's format is often the output's already and then there is no copy to write.
    float jitterX = 0.0f;
    float jitterY = 0.0f;
    JitterShift_Dx12* jitterShift = nullptr;

    if (route == VuNrRoute::BackendPass)
    {
        if (!EnsureTexture(_jittered, _jitteredKey, key, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS, "jittered"))
            return false;

        jitterShift = JitterShiftFor(_device);
    }

    if (jitterShift != nullptr)
    {
        ++_jitterFrame;
        jitterX = HaltonSequence((int) _jitterFrame, 2) - 0.5f;
        jitterY = HaltonSequence((int) _jitterFrame, 3) - 0.5f;
    }

    if (jitterShift != nullptr)
    {
        Transition(cmd, frame.color, pictureNow, kRead);
        pictureNow = kRead;

        if (!jitterShift->Dispatch(cmd, frame.color, _jittered, jitterX, jitterY, key.width, key.height))
        {
            LOG_WARN("Jitter shift: dispatch failed, no jitter this frame");
            jitterX = 0.0f;
            jitterY = 0.0f;
        }

        Transition(cmd, _jittered, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, inColor);
        color = _jittered;
    }
    else if (copyIn)
    {
        Transition(cmd, frame.color, pictureNow, D3D12_RESOURCE_STATE_COPY_SOURCE);
        pictureNow = D3D12_RESOURCE_STATE_COPY_SOURCE;
        cmd->CopyResource(_input, frame.color);
        Transition(cmd, _input, D3D12_RESOURCE_STATE_COPY_DEST, inColor);
        color = _input;
    }
    else
    {
        Transition(cmd, frame.color, pictureNow, inColor);
        pictureNow = inColor;
    }

    _params->Set(NVSDK_NGX_Parameter_Color, color);
    _params->Set(NVSDK_NGX_Parameter_Depth, frame.depth);
    _params->Set(NVSDK_NGX_Parameter_MotionVectors, frame.motion);
    _params->Set(NVSDK_NGX_Parameter_Output, _output);
    _params->Set(NVSDK_NGX_Parameter_Jitter_Offset_X, jitterX);
    _params->Set(NVSDK_NGX_Parameter_Jitter_Offset_Y, jitterY);
    _params->Set(NVSDK_NGX_Parameter_MV_Scale_X, 1.0f);
    _params->Set(NVSDK_NGX_Parameter_MV_Scale_Y, 1.0f);
    _params->Set(NVSDK_NGX_Parameter_Reset, frame.reset ? 1u : 0u);
    _params->Set(NVSDK_NGX_Parameter_Sharpness, 0.0f);
    _params->Set(NVSDK_NGX_Parameter_DLSS_Pre_Exposure, 1.0f);

    // From here the resources are where a game's would be at its call: in the states the backend takes them in.
    Transition(cmd, frame.motion, kRead, inMotion);
    Transition(cmd, depth, kRead, inDepth);
    Transition(cmd, _output, kWrite, inOutput);

    // DLSS-NR hangs off NVSDK_NGX_D3D12_EvaluateFeature around the upscaler, not off the feature: before it (Run before SR,
    // or the finished-picture capture of depth and motion) and after it (over Output). Not with the backend's pass off:
    // NR already ran on the picture itself, above, and would otherwise run twice.
    if (route != VuNrRoute::InPlace)
        DlssNr::EvaluateBeforeUpscale(cmd, _params);
    _params->Set(NVSDK_NGX_Parameter_Depth, depth);

    Transition(cmd, frame.motion, inMotion, fgMotion);
    Transition(cmd, depth, inDepth, fgDepth);
    Transition(cmd, _output, inOutput, fgOutput);

    // Back from a pause (the game called its own upscaler for a moment): frame generation's input changed, as when a
    // game switches features, which it takes as a short pause, not a new context
    if (_paused)
    {
        _paused = false;

        if (state.currentFG != nullptr && state.activeFgInput == FGInput::Upscaler)
        {
            state.fgChanged = true;
            state.clearCapturedHudlesses = true;
        }

        LOG_INFO("Virtual upscaler: resumed with the {} backend", _backendName);
    }

    state.currentFeature = _feature.get();

    // No pass asked for (NR off, or NR on the picture itself): the call is here for frame generation, which takes motion
    // and depth from it, not for an upscale. The picture is already full size and antialiased, so the backend's pass
    // (1.5-3 ms at 1440p-4K on a mid-range card, more with FSR 4 or XeSS off Intel) is skipped and the picture goes on as
    // the game made it, with NR's changes when NR ran on it above. The HUD fix still reads Output as its HUD-less picture (UpscaleEnd, recorded before
    // the backend's pass would run): the picture is copied there for it, as it will be presented.
    const bool passThrough = route != VuNrRoute::BackendPass;
    _backendPass.store(!passThrough, std::memory_order_relaxed);

    if (passThrough && cfg.FGHUDFix.value_or_default())
    {
        const auto colorNow = (color == _input || color == _jittered) ? inColor : pictureNow;
        Transition(cmd, color, colorNow, D3D12_RESOURCE_STATE_COPY_SOURCE);
        Transition(cmd, _output, fgOutput, D3D12_RESOURCE_STATE_COPY_DEST);
        cmd->CopyResource(_output, color);
        Transition(cmd, _output, D3D12_RESOURCE_STATE_COPY_DEST, fgOutput);
        Transition(cmd, color, D3D12_RESOURCE_STATE_COPY_SOURCE, colorNow);
    }

    UpscalerInputsDx12::UpscaleStart(cmd, _params, _feature.get());
    UpscalerInputsDx12::UpscaleEnd(cmd, _params, _feature.get());

    Transition(cmd, frame.motion, fgMotion, inMotion);
    Transition(cmd, depth, fgDepth, inDepth);
    Transition(cmd, _output, fgOutput, inOutput);

    bool evaluated = false;

    if (passThrough)
    {
        // FSR frame generation waits for the feature's first counted frame
        _feature->CountFrame();
    }
    else
    {
        ScopedSkipHeapCapture skip {};
        try
        {
            evaluated = _feature->Evaluate(cmd, _params);
        }
        catch (...)
        {
            LOG_ERROR("Virtual upscaler: the {} backend threw an exception while evaluating", _backendName);
            evaluated = false;
        }
    }

    _params->Set(NVSDK_NGX_Parameter_Depth, frame.depth);

    if (evaluated)
        DlssNr::EvaluateAfterUpscale(cmd, _params);

    if ((int) route != _routeLogged || _evaluations == 0)
    {
        _routeLogged = (int) route;
        LOG_INFO("Virtual upscaler: {}",
                 route == VuNrRoute::None    ? "Neural Rendering is off: the picture goes to frame generation as it is, "
                                               "no upscaler pass"
                 : route == VuNrRoute::InPlace ? "Neural Rendering runs on the game's picture; the backend only feeds "
                                                 "frame generation, no upscaler pass"
                 : cfg.DlssNrEnabled.value_or_default() ? "upscaler pass on, Neural Rendering on its output"
                                                        : "upscaler pass on (Neural Rendering is off)");
    }

    if (_evaluations < 8 || _evaluations % 300 == 0)
        LOG_INFO("Virtual upscaler: Evaluate {} -> {}, {} {}x{}, colour format {}{}, reset {}, depth reversed {}",
                 _evaluations,
                 route == VuNrRoute::None    ? "pass-through"
                 : route == VuNrRoute::InPlace ? (nrRan ? "no pass (NR in place)" : "no pass (NR did not run)")
                 : evaluated                 ? "true"
                                             : "false",
                 _backendName, key.width, key.height, (int) key.format,
                 jitterShift != nullptr ? " (jittered)" : (copyIn ? " (copied)" : ""), frame.reset,
                 _featureKey.depthReversed);

    ++_evaluations;

    if (jitterShift != nullptr && (_evaluations < 8 || _evaluations % 300 == 0))
        LOG_INFO("Jitter shift: {} x {} px", jitterX, jitterY);

    // The backend hands its inputs back in the states it took them in.
    Transition(cmd, frame.motion, inMotion, kRead);
    Transition(cmd, depth, inDepth, kRead);

    if (jitterShift != nullptr)
        Transition(cmd, _jittered, inColor, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    else if (copyIn)
        Transition(cmd, _input, inColor, D3D12_RESOURCE_STATE_COPY_DEST);

    if (evaluated)
    {
        Transition(cmd, _output, inOutput, D3D12_RESOURCE_STATE_COPY_SOURCE);
        Transition(cmd, frame.color, pictureNow, D3D12_RESOURCE_STATE_COPY_DEST);
        cmd->CopyResource(frame.color, _output);
        Transition(cmd, frame.color, D3D12_RESOURCE_STATE_COPY_DEST, frame.pictureState);
        Transition(cmd, _output, D3D12_RESOURCE_STATE_COPY_SOURCE, kWrite);
    }
    else
    {
        Transition(cmd, frame.color, pictureNow, frame.pictureState);
        Transition(cmd, _output, inOutput, kWrite);
    }

    return evaluated || passThrough;
}

} // namespace native
