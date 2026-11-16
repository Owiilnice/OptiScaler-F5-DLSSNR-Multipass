#include "pch.h"
#include "D3D11_Hooks.h"

#include <Util.h>
#include <Config.h>

#include <proxies/KernelBase_Proxy.h>

#include <dlssnr/DlssNr_ExposureScan_Dx11.h>
#include <resource_tracking/GenericDepth_Dx11.h>
#include <native/NativeLowLatency.h>

#include <wrapped/wrapped_swapchain.h>

#include <detours/detours.h>

#include <d3d11_4.h>
#include <d3d11on12.h>
#include <dxgi1_6.h>

#include <cmath>
#include <unordered_map>
#include <vector>

#include "Hook_Utils.h"
#include "FG_Hooks.h"
#include "DxgiFactory_Hooks.h"
#include "SwapchainCreateRule.h"

#pragma intrinsic(_ReturnAddress)

bool _skipDx11Create = false;

// DirectX
using PFN_CreateSamplerState = rewrite_signature<decltype(&ID3D11Device::CreateSamplerState)>::type;

using PFN_Release = rewrite_signature<decltype(&IUnknown::Release)>::type;

static PFN_D3D11_CREATE_DEVICE o_D3D11CreateDevice = nullptr;
static PFN_D3D11_CREATE_DEVICE_AND_SWAP_CHAIN o_D3D11CreateDeviceAndSwapChain = nullptr;
static PFN_CreateSamplerState o_CreateSamplerState = nullptr;
static PFN_D3D11ON12_CREATE_DEVICE o_D3D11On12CreateDevice = nullptr;
static PFN_Release o_D3D11DeviceRelease = nullptr;

static HRESULT hkCreateSamplerState(ID3D11Device* This, const D3D11_SAMPLER_DESC* pSamplerDesc,
                                    ID3D11SamplerState** ppSamplerState);

VALIDATE_HOOK(hkD3D11DeviceRelease, PFN_Release)
static ULONG hkD3D11DeviceRelease(IUnknown* device)
{
    if (State::Instance().currentD3D11Device == device)
    {
        device->AddRef();
        auto refCount = o_D3D11DeviceRelease(device);

        if (refCount == 1)
        {
            LOG_DEBUG("Set State::Instance().currentD3D11Device = nullptr, was: {:X}", (size_t) device);
            State::Instance().currentD3D11Device = nullptr;
            native::lowlatency::OnDeviceReleased(device);
        }
    }

    auto result = o_D3D11DeviceRelease(device);
    return result;
}

// The ini setting wins; unset, the game's quirk decides.
static bool ElevateD3D11FeatureLevel()
{
    if (Config::Instance()->D3D11FeatureLevelElevation.has_value())
        return Config::Instance()->D3D11FeatureLevelElevation.value();

    return !(State::Instance().gameQuirks & GameQuirk::SkipD3D11FeatureLevelElevation);
}

static inline D3D11_FILTER UpgradeToAF(D3D11_FILTER f)
{
    if (Config::Instance()->AnisotropySkipPointFilter.value_or_default() &&
        (f == D3D11_FILTER_MIN_MAG_MIP_POINT || f == D3D11_FILTER_COMPARISON_MIN_MAG_MIP_POINT ||
         f == D3D11_FILTER_MINIMUM_MIN_MAG_MIP_POINT || f == D3D11_FILTER_MAXIMUM_MIN_MAG_MIP_POINT ||
         f == D3D11_FILTER_MIN_MAG_LINEAR_MIP_POINT || f == D3D11_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT ||
         f == D3D11_FILTER_MINIMUM_MIN_MAG_LINEAR_MIP_POINT || f == D3D11_FILTER_MAXIMUM_MIN_MAG_LINEAR_MIP_POINT ||
         f == D3D11_FILTER_MIN_POINT_MAG_LINEAR_MIP_POINT ||
         f == D3D11_FILTER_COMPARISON_MIN_POINT_MAG_LINEAR_MIP_POINT ||
         f == D3D11_FILTER_MINIMUM_MIN_POINT_MAG_LINEAR_MIP_POINT ||
         f == D3D11_FILTER_MAXIMUM_MIN_POINT_MAG_LINEAR_MIP_POINT))
    {
        return f;
    }

    if (f >= D3D11_FILTER_COMPARISON_MIN_MAG_MIP_POINT && f <= D3D11_FILTER_COMPARISON_ANISOTROPIC)
    {
        return Config::Instance()->AnisotropyModifyComp.value_or_default() ? D3D11_FILTER_COMPARISON_ANISOTROPIC : f;
    }

    if (f >= D3D11_FILTER_MINIMUM_MIN_MAG_MIP_POINT && f <= D3D11_FILTER_MINIMUM_ANISOTROPIC)
    {
        return Config::Instance()->AnisotropyModifyMinMax.value_or_default() ? D3D11_FILTER_MINIMUM_ANISOTROPIC : f;
    }

    if (f >= D3D11_FILTER_MAXIMUM_MIN_MAG_MIP_POINT && f <= D3D11_FILTER_MAXIMUM_ANISOTROPIC)
    {
        return Config::Instance()->AnisotropyModifyMinMax.value_or_default() ? D3D11_FILTER_MAXIMUM_ANISOTROPIC : f;
    }

    return D3D11_FILTER_ANISOTROPIC;
}

// ---------------------------------------------------------------------------------------------------------------
// CbProbe: find the game's projection matrix in its D3D11 constant buffers.
//
// DLSS can only antialias with real per-frame sub-pixel jitter of the projection matrix; a game with no upscaler never
// hands one over. This probe only observes: it records every constant buffer write and reports the 4x4 windows whose
// entries change by a fraction of a unit between frames -- the signature of a jittered projection matrix. Nothing is
// modified, so it is safe to run with any settings.
// ---------------------------------------------------------------------------------------------------------------
using PFN_MapCb = HRESULT(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, ID3D11Resource*, UINT, D3D11_MAP, UINT,
                                             D3D11_MAPPED_SUBRESOURCE*);
using PFN_UnmapCb = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, ID3D11Resource*, UINT);
using PFN_UpdateSubCb = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, ID3D11Resource*, UINT, const D3D11_BOX*, const void*,
                                                 UINT, UINT);

static PFN_MapCb o_CbMap = nullptr;
static PFN_UnmapCb o_CbUnmap = nullptr;
static PFN_UpdateSubCb o_CbUpdateSub = nullptr;
static bool _cbProbeInstalled = false;

struct CbPending
{
    ID3D11Resource* resource;
    const void* data;
    UINT bytes;
};

static std::vector<CbPending> _cbPending;
static std::unordered_map<ID3D11Resource*, std::vector<float>> _cbPrev;
static std::unordered_map<ID3D11Resource*, UINT> _cbSizes;
static std::unordered_map<ID3D11Resource*, std::vector<uint8_t>> _cbHits;
static int _cbLogged = 0;
static int _cbSizesLogged = 0;

static bool CbIsConstantBuffer(ID3D11Resource* res, UINT* bytes)
{
    ID3D11Buffer* buffer = nullptr;

    if (res == nullptr || FAILED(res->QueryInterface(IID_PPV_ARGS(&buffer))) || buffer == nullptr)
        return false;

    D3D11_BUFFER_DESC desc {};
    buffer->GetDesc(&desc);
    buffer->Release();

    const bool constant = (desc.BindFlags & D3D11_BIND_CONSTANT_BUFFER) != 0 && desc.ByteWidth >= 64 &&
                          desc.ByteWidth <= 16384;

    if (constant && bytes != nullptr)
        *bytes = desc.ByteWidth;

    return constant;
}

static void CbAnalyze(ID3D11Resource* res, const void* data, UINT bytes)
{
    if (res == nullptr || data == nullptr || bytes < 64)
        return;

    const UINT count = bytes / 4;
    const float* cur = (const float*) data;

    if (_cbSizes.find(res) == _cbSizes.end())
    {
        _cbSizes[res] = bytes;

        if (++_cbSizesLogged <= 12)
            LOG_INFO("CbProbe: constant buffer {}B, first floats {:.4f} {:.4f} {:.4f} {:.4f}", bytes, cur[0], cur[1], cur[2],
                     cur[3]);
    }

    auto& prev = _cbPrev[res];

    if (prev.size() != count)
    {
        prev.assign(cur, cur + count);
        _cbHits[res].assign(count / 4, 0);
        return;
    }

    auto& hits = _cbHits[res];

    if (hits.size() != count / 4)
        hits.assign(count / 4, 0);

    // Most constant buffers are rewritten with the same contents; the scan is the expensive part.
    if (memcmp(prev.data(), cur, count * sizeof(float)) == 0)
        return;

    for (UINT off = 0; off + 16 <= count; off += 4)
    {
        int changed = 0;
        int firstIndex = -1;
        float maxDelta = 0.0f;

        for (int i = 0; i < 16; i++)
        {
            float d = fabsf(cur[off + i] - prev[off + i]);

            if (d > 1e-7f)
            {
                changed++;
                firstIndex = firstIndex < 0 ? i : firstIndex;
                maxDelta = d > maxDelta ? d : maxDelta;
            }
        }

        // A jittered projection matrix moves one or two entries by well under a unit, leaves the rest alone, and does it
        // again on the next frame. Requiring three frames in a row drops the one-off animation noise.
        uint8_t& hit = hits[off / 4];

        if (changed < 1 || changed > 4 || maxDelta >= 0.02f)
        {
            hit = 0;
            continue;
        }

        if (hit < 3 && ++hit == 3 && _cbLogged < 200)
        {
            LOG_INFO("CbProbe: {:p} {}B off {} changed {} first {} maxDelta {:.6f} | {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} "
                     "{:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f}",
                     (void*) res, bytes, off, changed, firstIndex, maxDelta, cur[off], cur[off + 1], cur[off + 2],
                     cur[off + 3], cur[off + 4], cur[off + 5], cur[off + 6], cur[off + 7], cur[off + 8], cur[off + 9],
                     cur[off + 10], cur[off + 11], cur[off + 12], cur[off + 13], cur[off + 14], cur[off + 15]);
            _cbLogged++;
        }
    }

    memcpy(prev.data(), cur, count * sizeof(float));
}

static HRESULT STDMETHODCALLTYPE hkCbMap(ID3D11DeviceContext* ctx, ID3D11Resource* res, UINT subResource,
                                         D3D11_MAP mapType, UINT flags, D3D11_MAPPED_SUBRESOURCE* mapped)
{
    const auto hr = o_CbMap(ctx, res, subResource, mapType, flags, mapped);

    UINT bytes = 0;

    if (SUCCEEDED(hr) && mapped != nullptr && mapped->pData != nullptr && CbIsConstantBuffer(res, &bytes))
    {
        // Several buffers can be mapped before the first one is unmapped, so keep them all. A buffer the game never
        // unmaps would grow the list for ever, hence the cap.
        if (_cbPending.size() >= 32)
            _cbPending.erase(_cbPending.begin());

        _cbPending.push_back({ res, mapped->pData, bytes });
    }

    return hr;
}

static void STDMETHODCALLTYPE hkCbUnmap(ID3D11DeviceContext* ctx, ID3D11Resource* res, UINT subResource)
{
    for (auto it = _cbPending.begin(); it != _cbPending.end(); ++it)
    {
        if (it->resource == res)
        {
            CbAnalyze(it->resource, it->data, it->bytes);
            _cbPending.erase(it);
            break;
        }
    }

    o_CbUnmap(ctx, res, subResource);
}

static void STDMETHODCALLTYPE hkCbUpdateSub(ID3D11DeviceContext* ctx, ID3D11Resource* res, UINT dstSub,
                                            const D3D11_BOX* box, const void* data, UINT rowPitch, UINT depthPitch)
{
    UINT bytes = 0;

    if (data != nullptr && CbIsConstantBuffer(res, &bytes))
    {
        // A constant buffer is one dimensional: front and back are byte offsets into it.
        const UINT written = box == nullptr ? bytes : (box->back > box->front ? box->back - box->front : 0);
        CbAnalyze(res, data, written < bytes ? written : bytes);
    }

    o_CbUpdateSub(ctx, res, dstSub, box, data, rowPitch, depthPitch);
}

// The detours sit in the context's own functions, the way the exposure scan hooks the device's resource creation: a COM
// object keeps its methods at fixed slots whatever interface it really implements.
static void InstallCbProbe(ID3D11Device* device)
{
    if (_cbProbeInstalled || device == nullptr)
        return;

    ID3D11DeviceContext* ctx = nullptr;
    device->GetImmediateContext(&ctx);

    if (ctx == nullptr)
        return;

    PVOID* table = *(PVOID**) ctx;

    o_CbMap = (PFN_MapCb) table[14];
    o_CbUnmap = (PFN_UnmapCb) table[15];
    o_CbUpdateSub = (PFN_UpdateSubCb) table[48];

    ctx->Release();

    DetourTransactionBegin();
    DetourUpdateThread(GetCurrentThread());
    DetourAttach(&(PVOID&) o_CbMap, hkCbMap);
    DetourAttach(&(PVOID&) o_CbUnmap, hkCbUnmap);
    DetourAttach(&(PVOID&) o_CbUpdateSub, hkCbUpdateSub);

    if (const auto result = DetourTransactionCommit(); result != NO_ERROR)
    {
        LOG_WARN("CbProbe: hooking the context failed ({:X})", (UINT) result);
        o_CbMap = nullptr;
        o_CbUnmap = nullptr;
        o_CbUpdateSub = nullptr;
        return;
    }

    _cbProbeInstalled = true;
    LOG_INFO("CbProbe: watching constant buffer writes (Map {}, Unmap {}, UpdateSubresource {})", 14, 15, 48);
}

static void HookToDeviceLocal(ID3D11Device* InDevice)
{
    InstallCbProbe(InDevice);

    // The depth finder hooks the immediate context the first time any D3D11 device is captured, whether or not this device's
    // own sampler hooks are already in (a second call with the same device is a no-op inside Install()).
    GenericDepthDx11::Install(InDevice);
    // The exposure scan's creation hooks, while the scan is wanted (dlssnr/DlssNr_ExposureScan_Dx11.h)
    DlssNr::ExposureScanDx11::Install(InDevice);

    if (o_CreateSamplerState != nullptr || InDevice == nullptr)
        return;

    LOG_DEBUG("Dx11");

    // Get the vtable pointer
    PVOID* pVTable = *(PVOID**) InDevice;

    o_D3D11DeviceRelease = (PFN_Release) pVTable[2];
    o_CreateSamplerState = (PFN_CreateSamplerState) pVTable[23];

    // Apply the detour
    if (o_CreateSamplerState != nullptr || o_D3D11DeviceRelease != nullptr)
    {
        DetourTransactionBegin();
        DetourUpdateThread(GetCurrentThread());

        if (o_D3D11DeviceRelease != nullptr)
            DetourAttach(&(PVOID&) o_D3D11DeviceRelease, hkD3D11DeviceRelease);

        if (o_CreateSamplerState != nullptr)
            DetourAttach(&(PVOID&) o_CreateSamplerState, hkCreateSamplerState);

        auto detourResult = DetourTransactionCommit();
        if (detourResult != NO_ERROR)
        {
            LOG_ERROR("Failed to hook ID3D11Device: {:X}", detourResult);
            o_CreateSamplerState = nullptr;
            o_D3D11DeviceRelease = nullptr;
        }
    }
}

VALIDATE_HOOK(hkD3D11On12CreateDevice, PFN_D3D11ON12_CREATE_DEVICE)
static HRESULT hkD3D11On12CreateDevice(IUnknown* pDevice, UINT Flags, const D3D_FEATURE_LEVEL* pFeatureLevels,
                                       UINT FeatureLevels, IUnknown* const* ppCommandQueues, UINT NumQueues,
                                       UINT NodeMask, ID3D11Device** ppDevice, ID3D11DeviceContext** ppImmediateContext,
                                       D3D_FEATURE_LEVEL* pChosenFeatureLevel)
{
    LOG_DEBUG("Caller: {}, Device: {:X}", Util::WhoIsTheCaller(_ReturnAddress()), (UINT64) pDevice);

#ifdef ENABLE_DEBUG_LAYER_DX11
    Flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

    bool rtss = false;

    std::vector<IUnknown*> copyCommandQueues;

    if (ppCommandQueues && NumQueues > 0)
        copyCommandQueues.assign(ppCommandQueues, ppCommandQueues + NumQueues);

    // Assuming RTSS is creating a D3D11on12 device, not sure why but sometimes RTSS tries to create
    // it's D3D11on12 device with old CommandQueue which results crash
    // I am changing it's CommandQueue with current swapchain's command queue
    if (State::Instance().currentCommandQueue != nullptr &&
        copyCommandQueues[0] != State::Instance().currentCommandQueue &&
        GetModuleHandle(L"RTSSHooks64.dll") != nullptr && pDevice == State::Instance().currentD3D12Device)
    {
        LOG_INFO("Replaced RTSS CommandQueue with correct one {0:X} -> {1:X}", (UINT64) *ppCommandQueues,
                 (UINT64) State::Instance().currentCommandQueue);

        copyCommandQueues[0] = State::Instance().currentCommandQueue;

        rtss = true;
    }

    HRESULT result = E_FAIL;
    {
        ScopedCreatingD3DDevice skipCreatingD3DDevice {};
        result = o_D3D11On12CreateDevice(pDevice, Flags, pFeatureLevels, FeatureLevels, copyCommandQueues.data(),
                                         static_cast<UINT>(copyCommandQueues.size()), NodeMask, ppDevice,
                                         ppImmediateContext, pChosenFeatureLevel);
    }

    if (result == S_OK && *ppDevice != nullptr && !rtss && State::Instance().currentD3D12Device == nullptr)
    {
        LOG_INFO("Device captured, D3D11Device: {0:X}", (UINT64) *ppDevice);
        HookToDeviceLocal(*ppDevice);
    }

    if (result == S_OK && *ppDevice != nullptr)
        State::Instance().d3d11Devices.push_back(*ppDevice);

    LOG_FUNC_RESULT(result);

    return result;
}

VALIDATE_HOOK(hkD3D11CreateDevice, PFN_D3D11_CREATE_DEVICE)
static HRESULT hkD3D11CreateDevice(IDXGIAdapter* pAdapter, D3D_DRIVER_TYPE DriverType, HMODULE Software, UINT Flags,
                                   const D3D_FEATURE_LEVEL* pFeatureLevels, UINT FeatureLevels, UINT SDKVersion,
                                   ID3D11Device** ppDevice, D3D_FEATURE_LEVEL* pFeatureLevel,
                                   ID3D11DeviceContext** ppImmediateContext)
{
    if (_skipDx11Create)
    {
        LOG_DEBUG("Skip");
        return o_D3D11CreateDevice(pAdapter, DriverType, Software, Flags, pFeatureLevels, FeatureLevels, SDKVersion,
                                   ppDevice, pFeatureLevel, ppImmediateContext);
    }

    LOG_DEBUG("Caller: {}", Util::WhoIsTheCaller(_ReturnAddress()));

#ifdef ENABLE_DEBUG_LAYER_DX11
    Flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

    DXGI_ADAPTER_DESC desc {};
    std::wstring szName;
    if (pAdapter != nullptr)
    {
        {
            ScopedSkipSpoofingGlobal skipSpoofingGlobal {};

            if (pAdapter->GetDesc(&desc) == S_OK)
            {
                szName = desc.Description;
                LOG_INFO("Adapter Desc: {}", wstring_to_string(szName));

                if (desc.VendorId == VendorId::Microsoft)
                {
                    _skipDx11Create = true;

                    HRESULT result;
                    {
                        ScopedSkipParentWrapping skipParentWrapping {};
                        ScopedCreatingD3DDevice skipCreatingD3DDevice {};
                        result =
                            o_D3D11CreateDevice(pAdapter, DriverType, Software, Flags, pFeatureLevels, FeatureLevels,
                                                SDKVersion, ppDevice, pFeatureLevel, ppImmediateContext);
                    }

                    _skipDx11Create = false;

                    return result;
                }
            }
        }
    }

    if (ElevateD3D11FeatureLevel())
    {
        static const D3D_FEATURE_LEVEL levels[] = {
            D3D_FEATURE_LEVEL_11_1,
        };

        D3D_FEATURE_LEVEL maxLevel = D3D_FEATURE_LEVEL_1_0_CORE;

        for (UINT i = 0; i < FeatureLevels; ++i)
        {
            maxLevel = std::max(maxLevel, pFeatureLevels[i]);
        }

        if (maxLevel == D3D_FEATURE_LEVEL_11_0)
        {
            LOG_INFO("Overriding D3D_FEATURE_LEVEL, "
                     "Game requested D3D_FEATURE_LEVEL_11_0, "
                     "we need D3D_FEATURE_LEVEL_11_1!");

            pFeatureLevels = levels;
            FeatureLevels = ARRAYSIZE(levels);
        }
    }

    _skipDx11Create = true;

    HRESULT result;
    {
        ScopedSkipParentWrapping skipParentWrapping {};
        result = o_D3D11CreateDevice(pAdapter, DriverType, Software, Flags, pFeatureLevels, FeatureLevels, SDKVersion,
                                     ppDevice, pFeatureLevel, ppImmediateContext);
    }

    _skipDx11Create = false;

    if (result == S_OK && *ppDevice != nullptr && State::Instance().currentD3D12Device == nullptr)
    {
        LOG_INFO("Device captured");
        HookToDeviceLocal(*ppDevice);
    }

    LOG_FUNC_RESULT(result);

    if (result == S_OK && ppDevice != nullptr && *ppDevice != nullptr)
        State::Instance().d3d11Devices.push_back(*ppDevice);

    return result;
}

VALIDATE_HOOK(hkD3D11CreateDeviceAndSwapChain, PFN_D3D11_CREATE_DEVICE_AND_SWAP_CHAIN)
static HRESULT hkD3D11CreateDeviceAndSwapChain(IDXGIAdapter* pAdapter, D3D_DRIVER_TYPE DriverType, HMODULE Software,
                                               UINT Flags, const D3D_FEATURE_LEVEL* pFeatureLevels, UINT FeatureLevels,
                                               UINT SDKVersion, const DXGI_SWAP_CHAIN_DESC* pSwapChainDesc,
                                               IDXGISwapChain** ppSwapChain, ID3D11Device** ppDevice,
                                               D3D_FEATURE_LEVEL* pFeatureLevel,
                                               ID3D11DeviceContext** ppImmediateContext)
{
    if (_skipDx11Create)
    {

        LOG_DEBUG("Skip");
        return o_D3D11CreateDeviceAndSwapChain(pAdapter, DriverType, Software, Flags, pFeatureLevels, FeatureLevels,
                                               SDKVersion, pSwapChainDesc, ppSwapChain, ppDevice, pFeatureLevel,
                                               ppImmediateContext);
    }

    LOG_DEBUG("Caller: {}", Util::WhoIsTheCaller(_ReturnAddress()));

#ifdef ENABLE_DEBUG_LAYER_DX11
    Flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

    DXGI_ADAPTER_DESC desc {};
    std::wstring szName;
    if (pAdapter != nullptr)
    {
        {
            ScopedSkipSpoofingGlobal skipSpoofingGlobal {};

            if (pAdapter->GetDesc(&desc) == S_OK)
            {
                szName = desc.Description;
                LOG_INFO("Adapter Desc: {}", wstring_to_string(szName));

                if (desc.VendorId == VendorId::Microsoft)
                {
                    _skipDx11Create = true;

                    HRESULT result;
                    {
                        ScopedSkipParentWrapping skipParentWrapping {};
                        ScopedCreatingD3DDevice skipCreatingD3DDevice {};
                        result = o_D3D11CreateDeviceAndSwapChain(pAdapter, DriverType, Software, Flags, pFeatureLevels,
                                                                 FeatureLevels, SDKVersion, pSwapChainDesc, ppSwapChain,
                                                                 ppDevice, pFeatureLevel, ppImmediateContext);
                    }

                    _skipDx11Create = false;

                    return result;
                }
            }
        }
    }

    static const D3D_FEATURE_LEVEL levels[] = { D3D_FEATURE_LEVEL_11_1 };

    if (ElevateD3D11FeatureLevel())
    {
        D3D_FEATURE_LEVEL maxLevel = D3D_FEATURE_LEVEL_1_0_CORE;

        for (UINT i = 0; i < FeatureLevels; ++i)
        {
            maxLevel = std::max(maxLevel, pFeatureLevels[i]);
        }

        if (maxLevel == D3D_FEATURE_LEVEL_11_0)
        {
            LOG_INFO("Overriding D3D_FEATURE_LEVEL, "
                     "Game requested D3D_FEATURE_LEVEL_11_0, "
                     "we need D3D_FEATURE_LEVEL_11_1!");

            pFeatureLevels = levels;
            FeatureLevels = ARRAYSIZE(levels);
        }
    }

    if (pSwapChainDesc != nullptr && pSwapChainDesc->BufferDesc.Height == 2 && pSwapChainDesc->BufferDesc.Width == 2)
    {
        LOG_WARN("Overlay call!");

        _skipDx11Create = true;
        State::Instance().skipParentWrapping = true;

        auto result = o_D3D11CreateDeviceAndSwapChain(pAdapter, DriverType, Software, Flags, pFeatureLevels,
                                                      FeatureLevels, SDKVersion, pSwapChainDesc, ppSwapChain, ppDevice,
                                                      pFeatureLevel, ppImmediateContext);

        State::Instance().skipParentWrapping = false;
        _skipDx11Create = false;

        return result;
    }

    DXGI_SWAP_CHAIN_DESC localDesc {};

    // For vsync override
    if (pSwapChainDesc != nullptr)
    {
        memcpy(&localDesc, pSwapChainDesc, sizeof(DXGI_SWAP_CHAIN_DESC));

        if (!pSwapChainDesc->Windowed)
        {
            LOG_INFO("Game is creating fullscreen swapchain, disabled V-Sync overrides");
            Config::Instance()->OverrideVsync.set_volatile_value(false);
        }

        if (Config::Instance()->OverrideVsync.value_or_default())
        {
            localDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
            localDesc.Flags |= DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;

            if (localDesc.BufferCount < 2)
                localDesc.BufferCount = 2;
        }
    }

    // HDR output (output/SdrToHdr_*): on the desc after the edits above. The device does not exist yet, and there is no factory
    // either: the display is found on a plain one by the window.
    SdrToHdr::Plan hdrPlan;

    // Another window while frame generation's swapchain lives (FGHooks::IsOtherWindowThanFG): a plain chain, no HDR
    // output, State left alone
    const bool otherWindow = pSwapChainDesc != nullptr && FGHooks::IsOtherWindowThanFG(pSwapChainDesc->OutputWindow);

    if (pSwapChainDesc != nullptr && !otherWindow)
    {
        SdrToHdr::CreateRequest hdrRequest;
        hdrRequest.format = localDesc.BufferDesc.Format;
        hdrRequest.windowed = localDesc.Windowed;
        hdrRequest.swapEffect = localDesc.SwapEffect;
        hdrRequest.bufferCount = localDesc.BufferCount;
        hdrRequest.flags = localDesc.Flags;
        hdrRequest.sampleCount = localDesc.SampleDesc.Count;
        hdrRequest.hasOriginal = true; // the game's own desc, before the V-Sync override
        hdrRequest.originalSwapEffect = pSwapChainDesc->SwapEffect;
        hdrRequest.originalBufferCount = pSwapChainDesc->BufferCount;
        hdrRequest.originalFlags = pSwapChainDesc->Flags;
        hdrRequest.flipModel = localDesc.SwapEffect == DXGI_SWAP_EFFECT_FLIP_DISCARD ||
                               localDesc.SwapEffect == DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
        hdrRequest.hwnd = localDesc.OutputWindow;
        hdrPlan = SdrToHdr::PrepareCreate(hdrRequest, localDesc.BufferDesc.Format);
    }

    _skipDx11Create = true;

    // An older (bitblt) chain is made flip model for HDR (the wrapper keeps telling the game what it asked for)
    if (hdrPlan.proxied)
        SdrToHdr::ApplySwapDesc(hdrPlan, localDesc.SwapEffect, localDesc.BufferCount, localDesc.Flags);

    // (nested creation hooks pass through while the chain being made is the HDR one)
    SdrToHdr::ScopedProxiedCreate hdrScope(hdrPlan.proxied);

    auto result = o_D3D11CreateDeviceAndSwapChain(pAdapter, DriverType, Software, Flags, pFeatureLevels, FeatureLevels,
                                                  SDKVersion, pSwapChainDesc != nullptr ? &localDesc : nullptr,
                                                  ppSwapChain, ppDevice, pFeatureLevel, ppImmediateContext);

    if (FAILED(result) && hdrPlan.proxied)
    {
        SdrToHdr::RestoreSwapDesc(hdrPlan, localDesc.SwapEffect, localDesc.BufferCount, localDesc.Flags);
        SdrToHdr::Unproxy(hdrPlan, localDesc.BufferDesc.Format, result);
        result = o_D3D11CreateDeviceAndSwapChain(pAdapter, DriverType, Software, Flags, pFeatureLevels, FeatureLevels,
                                                 SDKVersion, &localDesc, ppSwapChain, ppDevice, pFeatureLevel,
                                                 ppImmediateContext);
    }

    _skipDx11Create = false;

    if (SUCCEEDED(result) && !otherWindow && ppDevice != nullptr && *ppDevice != nullptr &&
        State::Instance().currentD3D12Device == nullptr)
    {
        LOG_INFO("Device captured");
        HookToDeviceLocal(*ppDevice);
    }

    if (SwapchainCreate::Created(result, ppSwapChain) && pSwapChainDesc != nullptr && ppDevice != nullptr &&
        *ppDevice != nullptr)
    {
        // check for SL proxy
        IDXGISwapChain* realSC = nullptr;
        if (!Util::CheckForRealObject(__FUNCTION__, *ppSwapChain, (IUnknown**) &realSC))
            realSC = *ppSwapChain;

        if (!otherWindow)
            State::Instance().currentRealSwapchain = realSC;

        IUnknown* realDevice = nullptr;
        if (!Util::CheckForRealObject(__FUNCTION__, *ppDevice, (IUnknown**) &realDevice))
            realDevice = *ppDevice;

        if (!otherWindow)
        {
            State::Instance().screenWidth = static_cast<float>(pSwapChainDesc->BufferDesc.Width);
            State::Instance().screenHeight = static_cast<float>(pSwapChainDesc->BufferDesc.Height);
        }

        LOG_DEBUG("Created new swapchain: {0:X}, hWnd: {1:X}", (UINT64) *ppSwapChain,
                  (UINT64) pSwapChainDesc->OutputWindow);

        // A nested hook (the DXGI factory's) wrapped it first. Asked of the chain itself: for another window that hook
        // leaves currentWrappedSwapchain alone, so comparing with it would wrap the wrapper again
        WrappedIDXGISwapChain4* nested = nullptr;

        if ((*ppSwapChain)->QueryInterface(IID_PPV_ARGS(&nested)) == S_OK)
        {
            LOG_DEBUG("Already wrapped by a nested hook, skipping wrapping");

            // ... without the plan: give it the plan
            if (hdrPlan.proxied)
                nested->AdoptHdrPlan(hdrPlan, localDesc.Flags);

            nested->Release();
        }
        else
        {
            *ppSwapChain = new WrappedIDXGISwapChain4(realSC, realDevice, pSwapChainDesc->OutputWindow,
                                                      localDesc.Flags, false, false, &hdrPlan);

            if (otherWindow)
            {
                static_cast<WrappedIDXGISwapChain4*>(*ppSwapChain)->MarkOtherWindow();
            }
            else
            {
                State::Instance().currentSwapchain = *ppSwapChain;
                State::Instance().currentWrappedSwapchain = *ppSwapChain;
                DxgiFactoryHooks::NoteCreateStatus(result, realSC);
            }

            LOG_DEBUG("Created new WrappedIDXGISwapChain4: {0:X}, pDevice: {1:X}", (UINT64) *ppSwapChain,
                      (UINT64) *ppDevice);
        }
    }

    if (SUCCEEDED(result) && ppDevice != nullptr && *ppDevice != nullptr)
        State::Instance().d3d11Devices.push_back(*ppDevice);

    LOG_FUNC_RESULT(result);
    return result;
}

VALIDATE_HOOK(hkCreateSamplerState, PFN_CreateSamplerState)
static HRESULT hkCreateSamplerState(ID3D11Device* This, const D3D11_SAMPLER_DESC* pSamplerDesc,
                                    ID3D11SamplerState** ppSamplerState)
{
    if (pSamplerDesc == nullptr || This == nullptr)
        return E_INVALIDARG;

    LOG_FUNC();

    D3D11_SAMPLER_DESC newDesc {};

    newDesc.AddressU = pSamplerDesc->AddressU;
    newDesc.AddressV = pSamplerDesc->AddressV;
    newDesc.AddressW = pSamplerDesc->AddressW;
    newDesc.ComparisonFunc = pSamplerDesc->ComparisonFunc;
    newDesc.BorderColor[0] = pSamplerDesc->BorderColor[0];
    newDesc.BorderColor[1] = pSamplerDesc->BorderColor[1];
    newDesc.BorderColor[2] = pSamplerDesc->BorderColor[2];
    newDesc.BorderColor[3] = pSamplerDesc->BorderColor[3];
    newDesc.MinLOD = pSamplerDesc->MinLOD;
    newDesc.MaxLOD = pSamplerDesc->MaxLOD;

    if (Config::Instance()->AnisotropyOverride.has_value())
    {
        LOG_DEBUG("Overriding {2:X} to anisotropic filtering {0} -> {1}", pSamplerDesc->MaxAnisotropy,
                  Config::Instance()->AnisotropyOverride.value(), (UINT) newDesc.Filter);

        newDesc.Filter = UpgradeToAF(pSamplerDesc->Filter);
        newDesc.MaxAnisotropy = Config::Instance()->AnisotropyOverride.value();
    }
    else
    {
        newDesc.Filter = pSamplerDesc->Filter;
        newDesc.MaxAnisotropy = pSamplerDesc->MaxAnisotropy;
    }

    newDesc.MipLODBias = pSamplerDesc->MipLODBias;

    if ((newDesc.MipLODBias < 0.0f && newDesc.MinLOD != newDesc.MaxLOD) ||
        Config::Instance()->MipmapBiasOverrideAll.value_or_default())
    {
        if (Config::Instance()->MipmapBiasOverride.has_value())
        {
            LOG_DEBUG("Overriding mipmap bias {0} -> {1}", pSamplerDesc->MipLODBias,
                      Config::Instance()->MipmapBiasOverride.value());

            if (Config::Instance()->MipmapBiasFixedOverride.value_or_default())
                newDesc.MipLODBias = Config::Instance()->MipmapBiasOverride.value();
            else if (Config::Instance()->MipmapBiasScaleOverride.value_or_default())
                newDesc.MipLODBias = newDesc.MipLODBias * Config::Instance()->MipmapBiasOverride.value();
            else
                newDesc.MipLODBias = newDesc.MipLODBias + Config::Instance()->MipmapBiasOverride.value();

            newDesc.MipLODBias = std::clamp(newDesc.MipLODBias, -16.0f, 15.99f);
        }

        if (State::Instance().lastMipBiasMax < newDesc.MipLODBias)
            State::Instance().lastMipBiasMax = newDesc.MipLODBias;

        if (State::Instance().lastMipBias > newDesc.MipLODBias)
            State::Instance().lastMipBias = newDesc.MipLODBias;
    }

    return o_CreateSamplerState(This, &newDesc, ppSamplerState);
}

void D3D11Hooks::HookToDevice(ID3D11Device* InDevice) { HookToDeviceLocal(InDevice); }

void D3D11Hooks::Hook(HMODULE dx11Module)
{
    if (o_D3D11CreateDevice != nullptr)
        return;

    LOG_DEBUG("");

    o_D3D11CreateDevice = (PFN_D3D11_CREATE_DEVICE) KernelBaseProxy::GetProcAddress_()(dx11Module, "D3D11CreateDevice");
    o_D3D11CreateDeviceAndSwapChain = (PFN_D3D11_CREATE_DEVICE_AND_SWAP_CHAIN) KernelBaseProxy::GetProcAddress_()(
        dx11Module, "D3D11CreateDeviceAndSwapChain");
    o_D3D11On12CreateDevice =
        (PFN_D3D11ON12_CREATE_DEVICE) KernelBaseProxy::GetProcAddress_()(dx11Module, "D3D11On12CreateDevice");

    if (o_D3D11CreateDevice != nullptr || o_D3D11On12CreateDevice != nullptr ||
        o_D3D11CreateDeviceAndSwapChain != nullptr)
    {
        LOG_DEBUG("Hooking D3D11CreateDevice methods");

        DetourTransactionBegin();
        DetourUpdateThread(GetCurrentThread());

        if (o_D3D11CreateDevice != nullptr)
            DetourAttach(&(PVOID&) o_D3D11CreateDevice, hkD3D11CreateDevice);

        if (o_D3D11On12CreateDevice != nullptr)
            DetourAttach(&(PVOID&) o_D3D11On12CreateDevice, hkD3D11On12CreateDevice);

        if (o_D3D11CreateDeviceAndSwapChain != nullptr)
            DetourAttach(&(PVOID&) o_D3D11CreateDeviceAndSwapChain, hkD3D11CreateDeviceAndSwapChain);

        auto detourResult = DetourTransactionCommit();
        if (detourResult != NO_ERROR)
        {
            LOG_ERROR("Failed to hook ID3D11Device: {:X}", detourResult);
            o_D3D11CreateDevice = nullptr;
            o_D3D11On12CreateDevice = nullptr;
            o_D3D11CreateDeviceAndSwapChain = nullptr;
        }
    }
}

void D3D11Hooks::Unhook()
{
    DetourTransactionBegin();
    DetourUpdateThread(GetCurrentThread());

    if (o_D3D11CreateDevice != nullptr)
        DetourDetach(&(PVOID&) o_D3D11CreateDevice, hkD3D11CreateDevice);

    if (o_D3D11DeviceRelease != nullptr)
        DetourDetach(&(PVOID&) o_D3D11DeviceRelease, hkD3D11DeviceRelease);

    if (o_D3D11On12CreateDevice != nullptr)
        DetourDetach(&(PVOID&) o_D3D11On12CreateDevice, hkD3D11On12CreateDevice);

    if (o_D3D11CreateDeviceAndSwapChain != nullptr)
        DetourDetach(&(PVOID&) o_D3D11CreateDeviceAndSwapChain, hkD3D11CreateDeviceAndSwapChain);

    if (o_CreateSamplerState != nullptr)
        DetourDetach(&(PVOID&) o_CreateSamplerState, hkCreateSamplerState);

    auto detourResult = DetourTransactionCommit();
    if (detourResult != NO_ERROR)
    {
        LOG_ERROR("Failed to unhook ID3D11Device: {:X}", detourResult);
    }
    else
    {
        o_D3D11CreateDevice = nullptr;
        o_CreateSamplerState = nullptr;
        o_D3D11DeviceRelease = nullptr;
        o_D3D11On12CreateDevice = nullptr;
        o_D3D11CreateDeviceAndSwapChain = nullptr;
    }
}

#pragma endregion
