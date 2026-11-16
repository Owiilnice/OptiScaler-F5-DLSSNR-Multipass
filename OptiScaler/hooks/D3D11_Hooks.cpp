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
#include <cstdio>
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
// CbProbe: find the game's projection matrix in the D3D11 buffers the game writes.
//
// DLSS can only antialias with real per-frame sub-pixel jitter of the projection matrix; a game with no upscaler never
// hands one over. ETS2 writes no constant buffer at all through Map/Unmap/UpdateSubresource, so the probe covers the
// other channels a D3D11.1 renderer can use: UpdateSubresource1, the deferred contexts, and plain buffers that are
// neither vertex nor index buffers. Nothing is modified, so it is safe to run with any settings.
// ---------------------------------------------------------------------------------------------------------------
using PFN_MapCb = HRESULT(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, ID3D11Resource*, UINT, D3D11_MAP, UINT,
                                             D3D11_MAPPED_SUBRESOURCE*);
using PFN_UnmapCb = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, ID3D11Resource*, UINT);
using PFN_UpdateSubCb = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, ID3D11Resource*, UINT, const D3D11_BOX*, const void*,
                                                 UINT, UINT);
using PFN_UpdateSub1Cb = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, ID3D11Resource*, UINT, const D3D11_BOX*,
                                                  const void*, UINT, UINT, UINT);
using PFN_CreateDeferredCb = HRESULT(STDMETHODCALLTYPE*)(ID3D11Device*, UINT, ID3D11DeviceContext**);

// One per concrete context class. A deferred context implements the same methods with different functions, and a hook
// cannot tell which one called it except through the context's own vtable, so the originals are keyed by that.
struct CbVtable
{
    PFN_MapCb map = nullptr;
    PFN_UnmapCb unmap = nullptr;
    PFN_UpdateSubCb updateSub = nullptr;
    PFN_UpdateSub1Cb updateSub1 = nullptr;
};

struct CbPending
{
    ID3D11Resource* resource;
    const void* data;
    UINT bytes;
};

static std::unordered_map<void*, CbVtable> _cbVtables;
static std::unordered_map<void*, PFN_CreateDeferredCb> _cbDeviceVtables;
static std::vector<CbPending> _cbPending;
static std::unordered_map<ID3D11Resource*, std::vector<float>> _cbPrev;
static std::unordered_map<ID3D11Resource*, UINT> _cbSizes;
static std::unordered_map<ID3D11Resource*, std::vector<uint8_t>> _cbHits;
static std::unordered_map<ID3D11Resource*, UINT> _cbWrites;
static int _cbLogged = 0;
static int _cbSizesLogged = 0;
static int _cbOtherLogged = 0;
static int _cbDeferredContexts = 0;
static long _cbMapCalls = 0;
static long _cbUpdateSub1Calls = 0;

static bool CbDescribe(ID3D11Resource* res, UINT* bytes, UINT* bindFlags)
{
    ID3D11Buffer* buffer = nullptr;

    if (res == nullptr || FAILED(res->QueryInterface(IID_PPV_ARGS(&buffer))) || buffer == nullptr)
        return false;

    D3D11_BUFFER_DESC desc {};
    buffer->GetDesc(&desc);
    buffer->Release();

    if (bytes != nullptr)
        *bytes = desc.ByteWidth;

    if (bindFlags != nullptr)
        *bindFlags = desc.BindFlags;

    return true;
}

// The game does not jitter its projection matrix, so looking for the movement finds nothing. Dump the whole buffer
// instead and pick the matrix out by its structure: a -1 in the last row or column, and the two scale terms sit in
// 0.3..4 for a normal field of view.
static void CbDump(ID3D11Resource* res, UINT bytes, const float* cur, const char* when)
{
    const UINT count = bytes / 4;
    LOG_INFO("CbProbe: dump {:p} {}B at {}, {} floats", (void*) res, bytes, when, count);

    char line[1200];

    for (UINT base = 0; base < count; base += 24)
    {
        int len = snprintf(line, sizeof(line), "+%4u:", base);

        for (UINT i = base; i < base + 24 && i < count; i++)
            len += snprintf(line + len, sizeof(line) - len, " %.5f", cur[i]);

        LOG_INFO("{}", line);
    }
}

static void CbAnalyze(ID3D11Resource* res, const void* data, UINT bytes, UINT bindFlags)
{
    if (res == nullptr || data == nullptr || bytes < 64)
        return;

    const UINT count = bytes / 4;
    const float* cur = (const float*) data;

    if ((bindFlags & D3D11_BIND_CONSTANT_BUFFER) == 0)
    {
        // Geometry is uploaded constantly and would drown the log, and one line per buffer is enough to spot a matrix.
        if ((bindFlags & (D3D11_BIND_VERTEX_BUFFER | D3D11_BIND_INDEX_BUFFER)) != 0 || _cbOtherLogged >= 40 ||
            _cbWrites[res] != 0)
            return;

        _cbWrites[res] = 1;
        _cbOtherLogged++;
        LOG_INFO("CbProbe: buffer {:p} {}B flags {:X}, first floats {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} "
                 "{:.5f}",
                 (void*) res, bytes, bindFlags, cur[0], cur[1], cur[2], cur[3], cur[4], cur[5], cur[6], cur[7]);
        return;
    }

    const UINT writes = ++_cbWrites[res];

    if (_cbSizes.find(res) == _cbSizes.end())
    {
        _cbSizes[res] = bytes;

        if (++_cbSizesLogged <= 12)
            LOG_INFO("CbProbe: constant buffer {}B, first floats {:.4f} {:.4f} {:.4f} {:.4f}", bytes, cur[0], cur[1], cur[2],
                     cur[3]);
    }

    if (writes == 1 || writes == 200 || writes == 1000 || writes == 2000 || writes == 3000)
    {
        char when[32];
        snprintf(when, sizeof(when), "write %u", writes);
        CbDump(res, bytes, cur, when);
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

static CbVtable* CbTableFor(ID3D11DeviceContext* ctx)
{
    if (ctx == nullptr)
        return nullptr;

    auto found = _cbVtables.find(*(void**) ctx);
    return found == _cbVtables.end() ? nullptr : &found->second;
}

static HRESULT STDMETHODCALLTYPE hkCbMap(ID3D11DeviceContext* ctx, ID3D11Resource* res, UINT subResource,
                                         D3D11_MAP mapType, UINT flags, D3D11_MAPPED_SUBRESOURCE* mapped)
{
    auto table = CbTableFor(ctx);

    if (table == nullptr)
        return E_FAIL;

    const auto hr = table->map(ctx, res, subResource, mapType, flags, mapped);

    UINT bytes = 0;
    UINT bindFlags = 0;

    if ((++_cbMapCalls % 5000) == 0)
        LOG_INFO("CbProbe: {} Map calls so far, {} distinct constant buffers, {} deferred contexts", _cbMapCalls,
                 _cbSizes.size(), _cbDeferredContexts);

    if (SUCCEEDED(hr) && mapped != nullptr && mapped->pData != nullptr && CbDescribe(res, &bytes, &bindFlags))
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
            UINT bytes = 0;
            UINT bindFlags = 0;

            if (CbDescribe(res, &bytes, &bindFlags))
                CbAnalyze(res, it->data, it->bytes, bindFlags);

            _cbPending.erase(it);
            break;
        }
    }

    if (auto table = CbTableFor(ctx); table != nullptr)
        table->unmap(ctx, res, subResource);
}

static void STDMETHODCALLTYPE hkCbUpdateSub(ID3D11DeviceContext* ctx, ID3D11Resource* res, UINT dstSub,
                                            const D3D11_BOX* box, const void* data, UINT rowPitch, UINT depthPitch)
{
    UINT bytes = 0;
    UINT bindFlags = 0;

    if (data != nullptr && CbDescribe(res, &bytes, &bindFlags))
    {
        // A buffer is one dimensional: front and back are byte offsets into it.
        const UINT written = box == nullptr ? bytes : (box->back > box->front ? box->back - box->front : 0);
        CbAnalyze(res, data, written < bytes ? written : bytes, bindFlags);
    }

    if (auto table = CbTableFor(ctx); table != nullptr)
        table->updateSub(ctx, res, dstSub, box, data, rowPitch, depthPitch);
}

static void STDMETHODCALLTYPE hkCbUpdateSub1(ID3D11DeviceContext* ctx, ID3D11Resource* res, UINT dstSub,
                                             const D3D11_BOX* box, const void* data, UINT rowPitch, UINT depthPitch,
                                             UINT copyFlags)
{
    UINT bytes = 0;
    UINT bindFlags = 0;

    if (++_cbUpdateSub1Calls % 5000 == 0)
        LOG_INFO("CbProbe: {} UpdateSubresource1 calls so far", _cbUpdateSub1Calls);

    if (data != nullptr && CbDescribe(res, &bytes, &bindFlags))
    {
        const UINT written = box == nullptr ? bytes : (box->back > box->front ? box->back - box->front : 0);
        CbAnalyze(res, data, written < bytes ? written : bytes, bindFlags);
    }

    if (auto table = CbTableFor(ctx); table != nullptr && table->updateSub1 != nullptr)
        table->updateSub1(ctx, res, dstSub, box, data, rowPitch, depthPitch, copyFlags);
}

// Slots 14..116 of one shared table: writing them in place reaches every context of this class, which is what we want,
// and avoids copying a table whose length depends on the interface the game asked for.
static void InstallCbHooksOnContext(ID3D11DeviceContext* ctx)
{
    if (ctx == nullptr)
        return;

    void** vt = *(void***) ctx;

    if (_cbVtables.find(vt) != _cbVtables.end())
        return;

    ID3D11DeviceContext1* ctx1 = nullptr;
    const bool hasContext1 = SUCCEEDED(ctx->QueryInterface(IID_PPV_ARGS(&ctx1)));

    if (ctx1 != nullptr)
        ctx1->Release();

    CbVtable table {};
    table.map = (PFN_MapCb) vt[14];
    table.unmap = (PFN_UnmapCb) vt[15];
    table.updateSub = (PFN_UpdateSubCb) vt[48];

    if (hasContext1)
        table.updateSub1 = (PFN_UpdateSub1Cb) vt[116];

    const size_t first = 14;
    const size_t last = hasContext1 ? 116 : 48;
    const size_t span = (last - first + 1) * sizeof(void*);
    DWORD oldProtect = 0;

    if (!VirtualProtect(&vt[first], span, PAGE_READWRITE, &oldProtect))
    {
        LOG_WARN("CbProbe: cannot write the context vtable");
        return;
    }

    // Registered before the slots are replaced: a hook can fire as soon as the first one is in place.
    _cbVtables[vt] = table;

    vt[14] = (void*) hkCbMap;
    vt[15] = (void*) hkCbUnmap;
    vt[48] = (void*) hkCbUpdateSub;

    if (hasContext1)
        vt[116] = (void*) hkCbUpdateSub1;

    VirtualProtect(&vt[first], span, oldProtect, &oldProtect);
    LOG_INFO("CbProbe: context hooked (Map {}, Unmap {}, UpdateSubresource {}, UpdateSubresource1 {})", 14, 15, 48,
             hasContext1 ? 116 : -1);
}

static HRESULT STDMETHODCALLTYPE hkCbCreateDeferred(ID3D11Device* device, UINT flags, ID3D11DeviceContext** ctx)
{
    auto found = _cbDeviceVtables.find(device == nullptr ? nullptr : *(void**) device);

    if (found == _cbDeviceVtables.end())
        return E_FAIL;

    const auto hr = found->second(device, flags, ctx);

    if (SUCCEEDED(hr) && ctx != nullptr && *ctx != nullptr)
    {
        ++_cbDeferredContexts;
        InstallCbHooksOnContext(*ctx);
        LOG_INFO("CbProbe: deferred context {:p}, {} so far", (void*) *ctx, _cbDeferredContexts);
    }

    return hr;
}

static void InstallCbProbe(ID3D11Device* device)
{
    if (device == nullptr)
        return;

    ID3D11DeviceContext* ctx = nullptr;
    device->GetImmediateContext(&ctx);

    if (ctx != nullptr)
    {
        InstallCbHooksOnContext(ctx);
        ctx->Release();
    }

    void** vt = *(void***) device;

    if (_cbDeviceVtables.find(vt) != _cbDeviceVtables.end())
        return;

    _cbDeviceVtables[vt] = (PFN_CreateDeferredCb) vt[27];

    DWORD oldProtect = 0;

    if (VirtualProtect(&vt[27], sizeof(void*), PAGE_READWRITE, &oldProtect))
    {
        vt[27] = (void*) hkCbCreateDeferred;
        VirtualProtect(&vt[27], sizeof(void*), oldProtect, &oldProtect);
        LOG_INFO("CbProbe: watching deferred context creation");
    }
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
