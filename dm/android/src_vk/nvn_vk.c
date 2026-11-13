// Vulkan graphics backend for Android: the nvn_mtl.h interface the NVN layer calls (on Apple it is nvn_mtl.m), replaying each submission's command list into a Vulkan command buffer.
// Shaders are the SPIR-V the Metal chain translates (dm/android/tools/spv_android.py: <key>.spv + <key>.vk.txt in DM_VK_DIR), every resource in set 0 with a binding of its own; here the vertex
// stage keeps set 0 and the fragment stage is moved to set 1 (its DescriptorSet decorations rewritten at load), so one pipeline layout holds both.
// Memory: Vulkan cannot use the game's memory pools in place (no VK_EXT_external_memory_host on Mali), so what a draw reads from them -- uniform buffers, vertex and index data, the shader's
// own constants -- is copied into the submission's ring buffer when the draw is recorded. The command list is walked in order on the CPU and its in-order writes (inline uniform updates,
// buffer copies and fills) are applied to guest memory where they stand, so each draw copies what the GPU would have read at that point. Textures are Vulkan images: packaged ones
// (block-linear in the pool) are deswizzled and uploaded at first use; render targets, copies and clears stay on the GPU. Every image lives in VK_IMAGE_LAYOUT_GENERAL, and passes and copies
// are separated by full memory barriers: simple and correct first, faster later.
// Event values, counter reports and texture-to-buffer copies reach guest memory when the GPU has finished the submission (the completion thread), as on the Metal backend.
// With a window (the app, app/front_android.c) presented frames are scaled onto a swapchain, keeping the game's aspect unless DM_SCREEN_FIT=stretch; DM_SHOW_FPS=1 draws a frame-rate
// counter, DM_FPS=30 holds every frame two vsyncs. Without one (the adb front end) nothing is shown: DM_VK_SHOTS=<frame>,<frame>,... writes those frames to DM_VK_SHOT_DIR as BMP files.
#include "../../gfx/nvn/nvn_dirty.h"
#include "hle.h"
#include "nvn_obj.h"
#include "nvn_mtl.h"
#include "../tex/nvn_deswizzle.h"
#define VK_USE_PLATFORM_ANDROID_KHR
#include <vulkan/vulkan.h>
#include <android/native_window.h>
#include <dlfcn.h>
#include <stdatomic.h>
#include <sys/stat.h>
#include <math.h>
#include <sched.h>
#include <sys/resource.h>

// Every vkAllocateMemory / vkFreeMemory of this file goes through these (vk19.31): the bytes live in all, for the memory lines. The Adreno driver held about 500 MB more GPU memory
// than the backend's own count (dumpsys gpu --gpumem: 1.07 GB for FINAL FANTASY RESONANCE against 530 MB of blocks and rings), outside the resident memory and against MemAvailable
static pthread_mutex_t g_dmmu = PTHREAD_MUTEX_INITIALIZER; static struct { uint64_t h, size; } g_dm[8192]; static _Atomic int64_t g_dm_live, g_dm_n;
static VkResult vk_alloc_counted(VkDevice d, const VkMemoryAllocateInfo *ai, const VkAllocationCallbacks *a, VkDeviceMemory *m) {
    VkResult r = vkAllocateMemory(d, ai, a, m); if (r != VK_SUCCESS) return r;
    uint64_t h = (uint64_t)(uintptr_t)*m; atomic_fetch_add(&g_dm_live, (int64_t)ai->allocationSize); atomic_fetch_add(&g_dm_n, 1);
    pthread_mutex_lock(&g_dmmu); for (uint32_t i = (uint32_t)(h * 0x9E3779B97F4A7C15ull >> 51), k = 0; k < 8192; k++, i = (i + 1) & 8191) if (!g_dm[i].h || g_dm[i].h == ~0ull) { g_dm[i].h = h; g_dm[i].size = ai->allocationSize; break; }
    pthread_mutex_unlock(&g_dmmu); return r;
}
static void vk_free_counted(VkDevice d, VkDeviceMemory m, const VkAllocationCallbacks *a) {
    uint64_t h = (uint64_t)(uintptr_t)m;
    if (h) { pthread_mutex_lock(&g_dmmu); for (uint32_t i = (uint32_t)(h * 0x9E3779B97F4A7C15ull >> 51), k = 0; k < 8192 && g_dm[i].h; k++, i = (i + 1) & 8191)
                 if (g_dm[i].h == h) { atomic_fetch_sub(&g_dm_live, (int64_t)g_dm[i].size); atomic_fetch_sub(&g_dm_n, 1); g_dm[i].h = ~0ull; break; }   // ~0: a tombstone
             pthread_mutex_unlock(&g_dmmu); }
    vkFreeMemory(d, m, a);
}
#define vkAllocateMemory vk_alloc_counted
#define vkFreeMemory vk_free_counted
static VkPhysicalDevice g_pd;   // (defined with the device below)
static int g_has_budget;   // VK_EXT_memory_budget: the driver's own count of this process's memory per heap
static uint64_t gpu_driver_usage(uint64_t *budget) {
    if (!g_has_budget) { if (budget) *budget = 0; return 0; }
    VkPhysicalDeviceMemoryBudgetPropertiesEXT bp = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 mp = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, &bp};
    vkGetPhysicalDeviceMemoryProperties2(g_pd, &mp);
    uint64_t u = 0, b = 0; for (uint32_t i = 0; i < mp.memoryProperties.memoryHeapCount; i++) { u += bp.heapUsage[i]; b += bp.heapBudget[i]; }
    if (budget) *budget = b; return u;
}
static _Atomic uint64_t g_fbs_bytes;   // the same-pass feedback snapshots' images (fb_snap_for), never freed
#define VKCHECK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) die("Vulkan: %s failed (%d)", #x, (int)r_); } while (0)
#define RED_ONCE(id, ...) do { static int once_; if (!once_++) LOG("[RED] vk: " __VA_ARGS__); } while (0)
// Timers of the per-frame statistics (vk19.38): the arm64 virtual counter read in user space, not clock_gettime (3.8% of the render thread in a profile: two calls for every
// descriptor set, draw and pipeline lookup). Statistics only: waits, fences and deadlines keep now_ns
// DM_VK_LEAN (vk19.62, read at start only): the counter converted with one 64x64->128 multiply by a factor taken once (1e9 * 2^64 / frequency), not two divisions. The mode
// is fixed at the first call, so a start and an end of one measurement always use the same conversion (the two differ by up to a nanosecond)
static inline uint64_t stat_ns(void) {
#ifdef __aarch64__
    static uint64_t f; static int lean = -1; static unsigned __int128 mul; uint64_t v; __asm__ volatile("mrs %0, cntvct_el0" : "=r"(v));
    int l = __atomic_load_n(&lean, __ATOMIC_ACQUIRE);
    if (l < 0) { uint64_t q; __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(q)); f = q ? q : 1; const char *e = getenv("DM_VK_LEAN");   // the first calls (any thread): all compute the same
                 mul = ((unsigned __int128)1000000000ull << 64) / f; l = !(e && atoi(e) == 0); __atomic_store_n(&lean, l, __ATOMIC_RELEASE); }   // on unless DM_VK_LEAN=0 (vk19.65)
    if (l) return (uint64_t)(((unsigned __int128)v * mul) >> 64);
    return v / f * 1000000000ull + v % f * 1000000000ull / f;
#else
    return now_ns();
#endif
}
// Lock waits (vk19.38): the time a lock of the backend's was waited for (a failed trylock, then the lock), per lock, in the per-600 line
enum { LW_RMU, LW_GWMU, LW_MU, LW_QMU, LW_BLKMU, LW_N };
static _Atomic uint64_t g_lw_ns[LW_N], g_lw_n[LW_N];
static inline void lk_(pthread_mutex_t *m, int k) { if (pthread_mutex_trylock(m)) { uint64_t t = stat_ns(); pthread_mutex_lock(m); atomic_fetch_add_explicit(&g_lw_ns[k], stat_ns() - t, memory_order_relaxed); atomic_fetch_add_explicit(&g_lw_n[k], 1, memory_order_relaxed); } }

// ---------- H0 counters (Harvest Moon heavy-scene plan v2, docs/hm_android_perf_plan.md; measurement only, nothing behaves differently) ----------
// Logged with the per-600 lines as 'vk: H0 ...' (h0_lines): composes by caller and reason, generation changes by cause, copy_in misses by caller x reason, the write log, the
// index pool of draws whose indirect arguments the GPU writes, extra rings fresh / kept / not kept, the GPU-written-range lock held by holder, the recording thread's faults,
// user / system time and run-delay
enum { H0C_VTX, H0C_UBO, H0C_SSBO, H0C_TEXEL, H0C_IND, H0C_COPY, H0C_IMG, H0C_OTHER, H0C_N };
static int g_h0_caller = H0C_OTHER;   // who asks gw_find next (set just before each call on the recording thread)
static uint64_t g_h0_cmp[2][H0C_N];   // composes: [0] over an earlier submission's entries only, [1] over this submission's (gw_compose's ordered copies)
enum { H0R_PAST16, H0R_PAST256, H0R_PASTMORE, H0R_BEFORE, H0R_MISALIGN, H0R_MULTI, H0R_EXACT, H0R_N };
static uint64_t g_h0_cmp_why[H0R_N], g_h0_cmp_pass, g_h0_cmp_detail;   // reasons (a read can have several); passes ended for them; detailed lines logged
static uint64_t g_h0_gen[3];   // generation changes: 0 a new submission, 1 an extra ring, 2 the write log full
static uint64_t g_h0_ci[8][8], g_h0_mrgen[2];   // copy_in misses by caller x reason; MR_GEN split: [0] an earlier submission's copy, [1] this submission's (ring change or log overflow)
static uint64_t g_h0_wscan[2], g_h0_wscan_n[2];   // write-log entries scanned by a lookup that found its key: [0] copy_in, [1] the index-range cache
static uint64_t g_h0_ic_rescan[2];   // index ranges scanned again with the same key: [0] after a new submission, [1] within this one (ring change, overflow or a write)
static uint64_t g_h0_sub_gen0;   // g_wgen at this submission's start
static uint64_t g_h0_ipd, g_h0_ipd_starts, g_h0_ipd_pools, g_h0_ipd_bytes, g_h0_ipd_copies;   // indexed draws with GPU-written arguments; distinct index starts and pools a frame (summed); their index-pool copies
static _Atomic uint64_t g_h0_xr_fresh, g_h0_xr_fresh_b, g_h0_xr_kept, g_h0_xr_nokeep[4];   // extra rings: fresh (count, bytes), taken from the kept ones; not kept: 0 mapping unknown, 1 pool full, 2 byte cap, 3 memory tight
static _Atomic uint64_t g_h0_gwh_ns[2], g_h0_gwh_n[2];   // g_gwmu held: [0] by the recording thread's sites, [1] by the completion thread's
static uint32_t g_h0_ctx_used; static uint64_t g_h0_look, g_h0_walk;   // lookups that walked the entries, and the entries walked   // contexts used in the window (bit per context)
// DM_VK_H0=1 turns the H0 measurements on (off by default since the merge into main): the lock-hold timing, the compose classification, the indirect arguments read back
// at completion (and the scan of their indices), the index-pool start counts, the side copies of chained ranges' guest bytes, the per-submission walk over the entries, and the
// 'vk: H0' lines. Off, only plain counter increments remain on the hot paths
static int h0_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_H0"); v = e && atoi(e) != 0; if (v) LOG("vk: DM_VK_H0=1: the H0 measurements and their lines are on"); } return v; }
#define H0_GWH_T0() uint64_t h0t_ = h0_on() ? stat_ns() : 0
// H0 sawtooth (hm6: GPU time per frame growing ~0.28 ms a frame, reset at the early trims): per 100 frames a line ('vk: H0 frame'), and a line when a quantity falls or a chain breaks
static int g_h0_cpass;   // the open pass was begun by gw_compose (its draws are the composed passes' draws)
static uint64_t g_h0_cp_draws, g_h0_cp_prims, g_h0_cp_vbytes, g_h0_cmp_bytes, g_h0_cmp_cur_bytes;   // draws in composed passes (direct: count x instances), their bound vertex bytes; bytes composed (all, this submission's parts)
static uint64_t g_h0_chain_n, g_h0_chain_b, g_h0_dsp_fresh;   // dispatch storage buffers bound from a running submission's GPU-written range (chained: no snapshot), their bytes; bound from guest memory
static uint64_t g_h0_gw_n, g_h0_gw_b, g_h0_gw_nmax, g_h0_gw_bmax;   // GPU-written entries at submission starts (summed, max), their bytes
static _Atomic uint64_t g_h0_ii_draws, g_h0_ii_inst, g_h0_ii_prims, g_h0_ii_max, g_h0_ii_cp_inst, g_h0_ii_fi, g_h0_ii_bv;   // fi / bv: GPU-written indirect draws with a first instance / vertex offset not 0
static uint64_t g_h0_dbi; static _Atomic uint64_t g_h0_past_i, g_h0_past_v;   // GPU-written indirect draws whose arguments read past the index window / the vertex copy   // direct (and CPU-read indirect) draws with a base instance not 0   // indirect draws read back at completion: draws, instances, count x instances, most instances in one, instances in composed passes
static _Atomic uint64_t g_h0_lost_b, g_h0_lost_n, g_h0_snap_skip;   // chained write-backs: guest bytes the CPU wrote while the submission ran, overwritten by the whole-range write-back; ranges checked; not checked (too big)
static uint8_t *g_h0_indp;   // ind_args: the ring bytes of GPU-written arguments
#define H0_GWH_END(k) do { if (h0t_) { atomic_fetch_add_explicit(&g_h0_gwh_ns[k], stat_ns() - h0t_, memory_order_relaxed); atomic_fetch_add_explicit(&g_h0_gwh_n[k], 1, memory_order_relaxed); } } while (0)
static VkInstance g_inst; static VkPhysicalDevice g_pd; static VkDevice g_dev; static VkQueue g_q; static uint32_t g_qf;
static VkPhysicalDeviceMemoryProperties g_memp; static VkPhysicalDeviceProperties g_props;
static pthread_mutex_t g_qmu = PTHREAD_MUTEX_INITIALIZER;   // the queue (submissions, the present readback)
static pthread_mutex_t g_rmu = PTHREAD_MUTEX_INITIALIZER;   // texture / sampler registrations and the image table (game threads register while the submit thread reads)

static uint32_t mem_type(uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < g_memp.memoryTypeCount; i++) if ((bits >> i & 1) && (g_memp.memoryTypes[i].propertyFlags & want) == want) return i;
    for (uint32_t i = 0; i < g_memp.memoryTypeCount; i++) if (bits >> i & 1) return i;
    die("Vulkan: no memory type for %#x", bits);
}
static VkDeviceMemory mem_alloc(VkMemoryRequirements r, VkMemoryPropertyFlags want) {
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, r.size, mem_type(r.memoryTypeBits, want)};
    VkDeviceMemory m; VKCHECK(vkAllocateMemory(g_dev, &ai, NULL, &m)); return m;
}
// Host-visible buffers (the rings, the stand-ins): the first HOST_VISIBLE | HOST_COHERENT type, which on the Adreno is uncached (write-combined). The CPU also reads the rings back
// (a dispatch's storage-range snapshot, the completion's compare of what the GPU changed, ~3.5 MB each a frame in FINAL FANTASY RESONANCE's castle): uncached reads are slow.
// DM_VK_RING_CACHED=1 takes a HOST_CACHED coherent type where the GPU has one (the Snapdragon's IO coherence). The type taken is logged once
static int ring_cached(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_RING_CACHED"); v = e && atoi(e) != 0; } return v; }
static VkBuffer buf_new(VkDeviceSize n, VkBufferUsageFlags u, VkDeviceMemory *m, void **map) {
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, n, u, VK_SHARING_MODE_EXCLUSIVE, 0, NULL};
    VkBuffer b; VKCHECK(vkCreateBuffer(g_dev, &bi, NULL, &b));
    VkMemoryRequirements r; vkGetBufferMemoryRequirements(g_dev, b, &r);
    VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (ring_cached()) for (uint32_t i = 0; i < g_memp.memoryTypeCount; i++) if ((r.memoryTypeBits >> i & 1) && (g_memp.memoryTypes[i].propertyFlags & (want | VK_MEMORY_PROPERTY_HOST_CACHED_BIT)) == (want | VK_MEMORY_PROPERTY_HOST_CACHED_BIT)) { want |= VK_MEMORY_PROPERTY_HOST_CACHED_BIT; break; }
    { static int logged; if (!logged++) { uint32_t t = mem_type(r.memoryTypeBits, want); LOG("vk: host-visible buffers in memory type %u (flags %#x%s); DM_VK_RING_CACHED=%d", t, g_memp.memoryTypes[t].propertyFlags,
                                                      g_memp.memoryTypes[t].propertyFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT ? ", cached" : ", uncached", ring_cached()); } }
    *m = mem_alloc(r, want);
    VKCHECK(vkBindBufferMemory(g_dev, b, *m, 0));
    if (map) VKCHECK(vkMapMemory(g_dev, *m, 0, VK_WHOLE_SIZE, 0, map));
    return b;
}
#define ALL_BUF_USAGE (VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | \
                       VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT | \
                       VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT)

static _Atomic uint64_t g_xr_peak; static uint64_t g_sub_up_bytes;   // the extra rings' peak since the memory watch's last line; upload bytes in this submission (reup_hold)
static _Atomic uint64_t g_big_bytes, g_extra_ring_bytes, g_ring0_bytes;   // memory of the big images and of the extra rings now held (the frame stats line)
// ---------- Image memory: 64 MB blocks, each with a list of its free ranges (first fit; a freed range joins its neighbours; a block with no image left is freed) ----------
// Thousands of textures would exceed the allocation count (Vulkan promises only 4096). Blocks used to be handed out in order and freed only when their last image was: a game
// streaming its textures in and out kept every block alive with a few images in it, and Super Mario Odyssey's opening reached 4 GB in 64 blocks and was killed for memory
typedef struct { VkDeviceSize off, size; } mfree;
typedef struct { VkDeviceMemory m; VkDeviceSize size; int live; uint32_t type; mfree *fr; int nfr, cfr; } mblock;
enum { NBLK = 256 };
static mblock g_blk[NBLK]; static int g_nblk; static pthread_mutex_t g_blkmu = PTHREAD_MUTEX_INITIALIZER;   // images are made on the submit thread and freed on the completion thread
static void fr_put(mblock *b, VkDeviceSize off, VkDeviceSize size) {   // return a range, joined with the free ranges next to it (the list is kept sorted)
    int k = 0; while (k < b->nfr && b->fr[k].off < off) k++;
    if (k > 0 && b->fr[k - 1].off + b->fr[k - 1].size == off) { b->fr[k - 1].size += size; if (k < b->nfr && off + size == b->fr[k].off) { b->fr[k - 1].size += b->fr[k].size; memmove(b->fr + k, b->fr + k + 1, (b->nfr - k - 1) * sizeof *b->fr); b->nfr--; } return; }
    if (k < b->nfr && off + size == b->fr[k].off) { b->fr[k].off = off; b->fr[k].size += size; return; }
    if (b->nfr == b->cfr) { b->cfr = b->cfr ? 2 * b->cfr : 16; b->fr = realloc(b->fr, b->cfr * sizeof *b->fr); }
    memmove(b->fr + k + 1, b->fr + k, (b->nfr - k) * sizeof *b->fr); b->fr[k] = (mfree){off, size}; b->nfr++;
}
static int fr_take(mblock *b, VkDeviceSize size, VkDeviceSize align, VkDeviceSize *off) {   // first fit
    for (int k = 0; k < b->nfr; k++) {
        VkDeviceSize o = (b->fr[k].off + align - 1) / align * align, end = b->fr[k].off + b->fr[k].size;
        if (o + size > end) continue;
        VkDeviceSize lo = b->fr[k].off, hi = o + size;
        memmove(b->fr + k, b->fr + k + 1, (b->nfr - k - 1) * sizeof *b->fr); b->nfr--;
        if (o > lo) fr_put(b, lo, o - lo);
        if (end > hi) fr_put(b, hi, end - hi);
        *off = o; return 1;
    }
    return 0;
}
static int img_mem_(VkMemoryRequirements r, VkDeviceMemory *m, VkDeviceSize *off);
static void blk_spare_want(uint32_t type);
static int img_mem(VkMemoryRequirements r, VkDeviceMemory *m, VkDeviceSize *off) { lk_(&g_blkmu, LW_BLKMU); int b = img_mem_(r, m, off); pthread_mutex_unlock(&g_blkmu); return b; }
// DM_VK_SPARE_BLOCK (vk-t9; the default is the profile's spare_block key, off unless it says on: +64 MB of device memory, measured on Tomodachi Life first): an empty image block is kept ready, so a texture that fits no block takes it instead of allocating 64 MB on the recording thread.
// The Adreno driver zero-fills a new block inside vkAllocateMemory (kgsl ioctls): Tomodachi Life's island profiles put ~0.5 G cycles (~0.4 s) on the recording thread for each
// of the two blocks its texture loads added, a stall at every new block. Now the dm-blkspare thread makes one when the blocks of that memory type have less than 16 MB of
// free room left, and the last block to empty stays as the spare instead of being freed (it was freed and made again as textures came and went). Not while memory is short
// (g_mem_low, or a trim in the last 30 s: none made) and let go under real pressure (g_mem_tight; blk_spare_trim from the trims). g_blk_new_rec / g_blk_spare_used / g_blk_spare_made: the memory line
static int g_mem_low, g_mem_tight;   // (defined with mem_watch below)
static uint64_t g_blk_new_rec, g_blk_spare_used, g_blk_spare_made, g_blk_spare_freed, g_blk_spare_dropped;
static _Atomic uint64_t g_last_trim_ns;   // the last trim's time (blk_spare_trim): no spare is made for 30 s after one, so a trim and a new spare do not take turns
static int spare_hold(void) { uint64_t t = atomic_load(&g_last_trim_ns); return g_mem_low || g_mem_tight || (t && now_ns() - t < 30000000000ull); }
#ifndef DM_GAME_SPARE_BLOCK
#define DM_GAME_SPARE_BLOCK 0
#endif
static int spare_block_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_SPARE_BLOCK"); v = e ? atoi(e) != 0 : DM_GAME_SPARE_BLOCK; LOG("vk: DM_VK_SPARE_BLOCK=%d: %s", v, v ? "an empty image block kept ready (made off the recording thread)" : "image blocks made when needed, freed when empty"); } return v; }
#ifndef DM_GAME_VK_POOL
#define DM_GAME_VK_POOL 0
#endif
// Profile key vk_pool (DM_VK_POOL=0/1 overrides): a shader whose texture handle the translator traced to no slot (kinds tex_pool / samp_pool) indexes the game's texture and
// sampler pools by ID; its pools go to the pool set (pool_shader / pool_set). Off: such pools are left as an unknown kind, as before (logged; the shader draws wrong)
static int pool_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_POOL"); v = e ? atoi(e) != 0 : DM_GAME_VK_POOL; if (v) LOG("vk: DM_VK_POOL=1: shaders indexing the texture and sampler pools by ID read the pool set"); } return v; }
// The pool set's texture arrays: IDs per class (DM_VK_POOL_CAP, at least 1024; the texture pool's capacity mtl_set_tex_pool_n reports above it is logged)
static uint32_t pool_cap(void) { static uint32_t c; if (!c) { const char *e = getenv("DM_VK_POOL_CAP"); c = e && atoi(e) >= 1024 ? (uint32_t)atoi(e) : 65536; } return c; }
static int blk_empty(const mblock *b) { return b->m && !b->live; }
static int img_mem_(VkMemoryRequirements r, VkDeviceMemory *m, VkDeviceSize *off) {
    uint32_t type = mem_type(r.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (r.size > (32u << 20)) { *m = mem_alloc(r, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT); *off = 0; atomic_fetch_add(&g_big_bytes, r.size); return -1; }   // big images get memory of their own
    for (int pass = 0; pass < 2; pass++)   // blocks in use first, the empty spare only when none has room (so it stays empty while it can)
        for (int i = 0; i < g_nblk; i++) { mblock *b = &g_blk[i]; if (b->m && b->type == type && blk_empty(b) == pass && fr_take(b, r.size, r.alignment, off)) {
            b->live++; *m = b->m;   // counted before the spare checks below: an emptied spare taken now is no longer one
            if (pass) g_blk_spare_used++;
            // a new spare only when the blocks' free room runs low -- also after the spare was taken: asking for one at every use made a block, and the image freed soon after
            // left two empty and one was freed: Tomodachi Life's island made and freed ~230 64 MB blocks in a minute (dm-blkspare 3% of the CPU), its textures coming and going
            if (spare_block_on()) { VkDeviceSize room = 0; for (int k = 0; k < g_nblk; k++) if (g_blk[k].m && g_blk[k].type == type) for (int j = 0; j < g_blk[k].nfr; j++) room += g_blk[k].fr[j].size;
                                    if (room < (16u << 20)) blk_spare_want(type); }
            return i; } }
    int i = 0; while (i < g_nblk && g_blk[i].m) i++;
    if (i == NBLK) die("Vulkan: image memory blocks exhausted");
    if (i == g_nblk) g_nblk++;
    mblock *b = &g_blk[i];
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, 64u << 20, type};
    VKCHECK(vkAllocateMemory(g_dev, &ai, NULL, &b->m)); g_blk_new_rec++;
    b->size = 64u << 20; b->live = 0; b->type = type; b->nfr = 0; fr_put(b, 0, b->size);
    if (!fr_take(b, r.size, r.alignment, off)) die("Vulkan: an image of %llu bytes does not fit a new memory block", (unsigned long long)r.size);
    b->live++; *m = b->m;
    if (spare_block_on()) blk_spare_want(type);
    return i;
}
static void img_mem_free(int blk, VkDeviceMemory m, VkDeviceSize off, VkDeviceSize size) {
    if (blk < 0) { vkFreeMemory(g_dev, m, NULL); atomic_fetch_sub(&g_big_bytes, size); return; }
    lk_(&g_blkmu, LW_BLKMU);
    mblock *b = &g_blk[blk]; fr_put(b, off, size);
    if (!--b->live) {   // empty: kept as the spare when it is the only empty block of its type (DM_VK_SPARE_BLOCK) and no spare would be made now (spare_hold)
        int keep = spare_block_on() && !spare_hold();
        for (int k = 0; keep && k < g_nblk; k++) if (k != blk && blk_empty(&g_blk[k]) && g_blk[k].type == b->type) keep = 0;
        if (!keep) { vkFreeMemory(g_dev, b->m, NULL); b->m = VK_NULL_HANDLE; b->nfr = 0; }
    }
    pthread_mutex_unlock(&g_blkmu);
}
// The spare's maker: one thread, woken by blk_spare_want (under g_blkmu); the 64 MB allocation runs with the lock released
static pthread_cond_t g_spare_cv = PTHREAD_COND_INITIALIZER; static int g_spare_want = -1, g_spare_busy;
static void *blk_spare_thread(void *arg) {
    (void)arg; pthread_setname_np("dm-blkspare");
    lk_(&g_blkmu, LW_BLKMU);
    for (;;) {
        while (g_spare_want < 0) pthread_cond_wait(&g_spare_cv, &g_blkmu);
        uint32_t type = (uint32_t)g_spare_want; g_spare_want = -1;
        int have = 0; for (int k = 0; k < g_nblk; k++) if (blk_empty(&g_blk[k]) && g_blk[k].type == type) have = 1;
        if (have || spare_hold()) continue;
        g_spare_busy = 1; pthread_mutex_unlock(&g_blkmu);
        VkDeviceMemory mm = VK_NULL_HANDLE; VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, 64u << 20, type};
        VkResult rr = vkAllocateMemory(g_dev, &ai, NULL, &mm);
        lk_(&g_blkmu, LW_BLKMU); g_spare_busy = 0;
        if (rr != VK_SUCCESS) { static int nl; if (nl++ < 4) LOG("vk: spare image block not made (%d)", (int)rr); continue; }
        // while it was made, the recording thread may have made its own block or a free may have left one empty, or memory became short: then it is not needed
        have = 0; for (int k = 0; k < g_nblk; k++) if (blk_empty(&g_blk[k]) && g_blk[k].type == type) have = 1;
        int i = 0; while (i < g_nblk && g_blk[i].m) i++;
        if (have || spare_hold() || i == NBLK) { pthread_mutex_unlock(&g_blkmu); vkFreeMemory(g_dev, mm, NULL); g_blk_spare_dropped++; lk_(&g_blkmu, LW_BLKMU); continue; }
        if (i == g_nblk) g_nblk++;
        mblock *b = &g_blk[i]; b->m = mm; b->size = 64u << 20; b->live = 0; b->type = type; b->nfr = 0; fr_put(b, 0, b->size); g_blk_spare_made++;
    }
    return NULL;
}
static void blk_spare_want(uint32_t type) {   // under g_blkmu
    static int started; if (spare_hold() || g_spare_busy) return;
    for (int k = 0; k < g_nblk; k++) if (blk_empty(&g_blk[k]) && g_blk[k].type == type) return;
    if (!started) { pthread_t th; if (pthread_create(&th, NULL, blk_spare_thread, NULL)) return; pthread_detach(th); started = 1; }
    g_spare_want = (int)type; pthread_cond_signal(&g_spare_cv);
}
static void blk_spare_trim(void) {   // the trims: empty blocks go back to the system
    atomic_store(&g_last_trim_ns, now_ns());
    lk_(&g_blkmu, LW_BLKMU);
    for (int k = 0; k < g_nblk; k++) if (blk_empty(&g_blk[k])) { vkFreeMemory(g_dev, g_blk[k].m, NULL); g_blk[k].m = VK_NULL_HANDLE; g_blk[k].nfr = 0; g_blk_spare_freed++; }
    pthread_mutex_unlock(&g_blkmu);
}

// ---------- Submission contexts: command buffers, ring buffer, descriptor pool, the writes that wait for completion ----------
#define NCTX_MAX 32
// Submission contexts (DM_VK_CTX=<n>, 4..32, default 20). 12 was ~1.3 frames of Unreal's ~9 submissions, chosen while frames were GPU-bound ("24 bought nothing"). With the
// GPU at 37 ms of a 73 ms frame (FINAL FANTASY RESONANCE's castle, vk19.8) the CPU waited 26 ms a frame for a context whose ring a later submission of the same frame still read
// (its hold): CPU and GPU took turns. More contexts let the CPU record the next frame while the GPU runs this one; each costs a ring (4 MB up, following use)
static int NCTX = 20;
// earlier: NCTX 12   // ~1.3 frames of Unreal's ~9 submissions (24 bought nothing on a GPU-bound frame and cost ring memory): with 8 the CPU could not get one frame ahead, and reusing a context whose ring a later submission still read (its hold) drained the GPU first: CPU and GPU took turns (recording 12.7 + waiting 12.5 ms a frame) // submissions in flight: Unreal submits several command buffers a frame, and with 3 the CPU could not record a frame ahead of the GPU
#define NOQ 4096   // occlusion queries per submission
#define NTQ 1024   // GPU timestamps per submission (512 regions)
// Ring sizes: each context's own ring starts at RING_MIN and follows what its submissions use (ring_fit: grown to the peak, up to RING_MAX; halved after a stretch that used less
// than a quarter; back to RING_MIN on a low-memory signal); a submission that outgrows it gets extra rings of the size it needs, freed when it is done. Eight fixed 48 MB rings
// held 384 MB of host-visible memory whatever the game needed, and with FINAL FANTASY RESONANCE open HyperOS killed other apps for memory
#define RING_MIN (4u << 20)
#define RING_MAX (64u << 20)
#define RING_XMAX (256u << 20)
// All contexts' own rings together (DM_VK_RING_CAP_MB, default 96; it was 160): past it a ring does not grow, and a submission that outgrows it takes extra rings (freed when it is
// done). The run that loaded a save into FINAL FANTASY RESONANCE had its rings at the 160 MB cap plus 16 MB extra when the low-memory killer ended it
static uint64_t ring_cap(void) { static uint64_t v; if (!v) { const char *e = getenv("DM_VK_RING_CAP_MB"); long mb = e && atol(e) > 0 ? atol(e) : 96; v = (uint64_t)mb << 20; LOG("vk: rings capped at %ld MB in all (DM_VK_RING_CAP_MB)", mb); } return v; }
#define RING_CAP ring_cap()
typedef struct { int kind; void *dst; uint32_t mode, v; uint64_t addr; uint32_t type; const uint8_t *src; size_t n; } post_t;   // kind 5: n bytes from src where they differ from the snapshot at addr (4-byte words); kind 6: v rows of type bytes, mode apart
static _Atomic uint64_t g_gw_wb_ended;   // GPU-written entries ended by their dispatch write-back (ctx_complete kind 5; the descriptor-set line)
static uint64_t g_wb_words, g_wb_bytes;   // dispatch write-backs: words the GPU changed and written back / bytes of the ranges looked at (completion thread)   // kind 1 event, 2 counter, 3 copy ring → guest, 4 samples-passed report (src: n query indices; n == SIZE_MAX: unknown, reported visible)
typedef struct vtex vtex;
typedef struct {
    VkCommandPool pool; VkCommandBuffer cb, up; VkFence fence; VkDescriptorPool dp;
    VkCommandBuffer prim; VkCommandBuffer *sec; int nsec, csec;   // DM_VK_STORE_NONE: the primary while a pass records into a secondary (sec[nsec - 1]); the pool's secondaries
    VkBuffer ring; VkDeviceMemory ringm; uint8_t *ringp; VkDeviceSize used;
    post_t *post; int npost, cpost;
    uint64_t fence_val; int up_used;
    VkQueryPool qp;   // GPU timestamps at the start and the end of the submission
    VkQueryPool oq; uint32_t noq; int oq_init;
    VkQueryPool tq; uint32_t ntq, ntq_prev; int tq_init; uint8_t tkind[NTQ / 2]; uint64_t tsig[NTQ / 2]; uint32_t tdraws[NTQ / 2];   // GPU timestamps around each pass / dispatch / copy (gpu_mark): a pair per region, its kind   // occlusion queries of this submission (one per draw made while NVN's samples-passed counter counts), reset at its start
    VkFramebuffer *fb; int nfb, cfb;   // render passes' framebuffers (no dynamic rendering), destroyed when the submission is done
    VkBuffer ring0; VkDeviceMemory ringm0; uint8_t *ringp0;   // the context's own ring; ring / ringp move to extra rings when a submission outgrows it (loading screens upload hundreds of MB)
    VkDeviceSize ring0_size, ring_size, total, last_total, win_max; uint32_t nwin; uint64_t trim_seen; uint64_t h05_last;   // h05_last: the submission that last took it (DM_VK_RING_REUSE)   // ring_size: of the ring in use; total: bytes this submission took from rings
    struct { VkBuffer b; VkDeviceMemory m; VkDeviceSize size; } *extra; int nextra;
    VkDescriptorPool *dps; int ndp, idp;   // descriptor pools, chained as a submission needs more (reset together at its start)
    int plog;   // a pass-logged frame recorded into it: its small write-backs are logged at completion (the values a dispatch computed, e.g. Unreal's exposure)
    VkBufferView *bv; int nbv, cbv;   // texel buffer views of this submission's descriptors (write_set), destroyed when it is done
    int reset_done;   // its command and descriptor pools were reset by the completion thread (ctx_complete), off the submit thread (each reset took ~9% of it)
    int resetting;   // the completion thread is resetting its pools after it signalled the fence (DM_VK_EARLY_DONE, under g_mu): not taken until done
    uint64_t hold;   // the last later submission that reads this one's ring (gw_find): the ring is not reused, nor its extra rings freed, before that one is done too
    // The work recorded into it (draws and dispatches), the last TRN kept and the heaviest noted: dumped when the GPU is lost (trace_dump), to name what hung it
    struct trace { char kind, gpu_ind; char a[17], b[17]; const char *pa, *pb; uint32_t n0, n1, n2, zero_tb; } tr[48]; uint32_t ntr, ndraw, ndisp, max_groups; struct trace heavy;
    const uint8_t *h0ip[256]; uint8_t h0ix[256]; int nh0ip;
    const uint8_t *h0ipp[256]; uint32_t h0iw[256], h0vmax[256]; uint8_t h0isz[256];   // H0 (dark vegetation): the index copy (its bytes, window) and the vertices copied, to check the GPU's arguments against at completion   // H0: GPU-written indirect arguments of this submission's draws (read back at completion); bit 0 indexed, bit 1 in a pass a compose began   // pa / pb: the shader keys themselves (DM_VK_LEAN; shaders are never freed), else copied into a / b
} ctx_t;
enum { TRN = 48 };
static _Atomic uint64_t g_gpu_ns, g_gpu_cum;   // g_gpu_cum: never reset (the long-frame lines)
static _Atomic uint64_t g_tex_made;   // textures made (vtex), for the memory watch's per-second line
static _Atomic uint64_t g_wait_cum, g_gw_wait_cum, g_up_cum, g_xring_n;   // cumulative: context waits, waits on GPU-written data, upload bytes, extra rings made (long frames)
static ctx_t g_ctx[NCTX_MAX]; static ctx_t *C; static int g_ci;
static uint64_t g_fence, g_done;
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t g_cv = PTHREAD_COND_INITIALIZER;
static int g_pend[NCTX_MAX], g_npend;   // contexts submitted and not yet completed, in order (under g_mu)

static uint64_t g_wgen;   // a new generation when the ring changes or the write log overflows: the per-submission copies are reused only within one (and only where no write since overlaps them, copy_in)
static uint32_t g_nwlog;   // writes logged in this generation (wlog_add)
static int g_lean_tr;   // DM_VK_LEAN for the submission (tog_latch): trace_add keeps pointers
static uint64_t g_bgen[6], g_bgen_next = 1;   // a stage's binding generation: changes when its texture, image, sampler or storage-buffer bindings change (DM_VK_SEG_CACHE, DM_VK_FB_FAST)
static _Atomic uint64_t g_cpr_ns, g_cpr_n, g_cpr_bufs, g_cpr_rec;   // command-pool resets: dm-complete's time and count, those done buffer by buffer, those left to the recording thread (vk19.64)
static int pool_reset_mode(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_POOL_RESET"); v = e ? atoi(e) : 0; if (v) LOG("vk: DM_VK_POOL_RESET=%d: command buffers reset one by one by dm-complete", v); } return v; }
static uint64_t g_tex_draws_win;   // draws recorded in the 600-frame window (trace_add; tex_get calls per draw, vk19.59)
static int zc_mode(void); static int zc_find(uint64_t addr, uint64_t n, VkBuffer *b, VkDeviceSize *off); static void zc_reap(uint64_t done);   // zero-copy pools (DM_VK_ZEROCOPY)
static int g_zc_dev; static uint64_t g_zc_bytes, g_zc_hits, g_zc_miss_wlog;
// Extra rings made and freed every few frames were fresh device memory each time (vkAllocateMemory, mapped: page faults as the CPU first writes them). Freed ones are kept (at most
// 4, 64 MB in all) and taken again by the next submission that outgrows its ring; the mapping of each live one is remembered by its memory (xring_*)
// DM_VK_RING_REUSE (Harvest Moon heavy scene, H0.5 of docs/hm_android_perf_plan.md; off by default). The pprof of the heavy scene: 30% of the recording thread in vkAllocateMemory
// (kgsl_sharedmem_alloc) for fresh extra rings and most of dm-complete in freeing them again, every frame: the frame's ~110-150 MB of copies outgrew the 4 MB rings of 20 contexts
// taken in turn, and the keep pool held 8 rings / 64 MB. With it on: (a) ctx_pick takes the most recently freed context (whose completion work is done first), so a game with
// one submission a frame works in 2-3 contexts; (b) ring_fit grows a hot context's ring to the largest power of two under the cap, the cap counting only bytes above each context's
// RING_MIN, and contexts unused for 300 submissions go back to RING_MIN; (c) the keep pool holds up to 32 extra rings, DM_VK_XRING_KEEP_MB 256 by default; (d) a submission's first
// extra ring is sized from the recent overflow (x 1.25, in 32 MB steps), so one ring usually takes the rest of a submission. Nothing changes in what is copied or bound
static int ring_reuse_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_RING_REUSE"); v = e && atoi(e) != 0; if (v) LOG("vk: DM_VK_RING_REUSE=1: recent contexts first, rings grown under the cap, up to 32 extra rings kept (DM_VK_XRING_KEEP_MB), extra rings sized from the overflow"); } return v; }
static VkDeviceSize g_h05_over;   // the recent overflow past a context's own ring (bytes; decays by 1/8 a submission)
static uint64_t g_h05_shrunk, g_h05_sized;   // rings of idle contexts given back; extra rings sized from the overflow
enum { NXC = 32 };
static struct { VkBuffer b; VkDeviceMemory m; void *p; VkDeviceSize size; } g_xc[NXC]; static int g_nxc; static VkDeviceSize g_xc_bytes;
// DM_VK_XRING_KEEP_MB (vk19.40, default 64; was 4 rings and 32 MB): the extra rings kept for reuse. With the rings at their 96 MB cap, the dungeon's submissions took about two
// extra rings a frame (48-64 MB live), and those the 32 MB could not keep were fresh mappings each time, every page faulted on its first write (~2,500 faults a frame left after
// the trims stopped). Kept rings are let go under real pressure (g_mem_tight) and by the trims. g_xr_fresh / g_ring_new: fresh extra rings and context rings made again (the per-600 line)
static uint64_t g_xr_fresh, g_xr_fresh_bytes, g_ring_new, g_ring_new_bytes;
static VkDeviceSize xring_keep(void) { static VkDeviceSize v; if (!v) { const char *e = getenv("DM_VK_XRING_KEEP_MB"); long mb = e && atol(e) >= 0 ? atol(e) : ring_reuse_on() ? 256 : 64; v = ((VkDeviceSize)mb << 20) | 1; } return v & ~(VkDeviceSize)1; }
static int g_mem_low;   // the system is short of memory (mem_watch): nothing is kept for reuse
static int g_mem_tight;   // real pressure (a burst, or under half the low mark): extra rings are not kept, rings go back to their minimum (vk19.38)
static int g_mem_real_low;   // MemAvailable under DM_VK_MEM_LOW_MB (or the budget passed): with DM_VK_RING_REUSE, the only state that lets kept rings go (an early trim's burst does not)
static _Atomic uint64_t g_ring_trim_gen;
static struct { VkDeviceMemory m; void *p; } g_xmap[256]; static int g_nxmap; static pthread_mutex_t g_xcmu = PTHREAD_MUTEX_INITIALIZER; static uint64_t g_xc_reused;
static VkBuffer xring_get(VkDeviceSize want, VkDeviceSize *size, VkDeviceMemory *m, uint8_t **p) {
    pthread_mutex_lock(&g_xcmu); int best = -1;
    for (int i = 0; i < g_nxc; i++) if (g_xc[i].size >= want && (best < 0 || g_xc[i].size < g_xc[best].size)) best = i;
    if (best >= 0) { VkBuffer b = g_xc[best].b; *m = g_xc[best].m; *p = g_xc[best].p; *size = g_xc[best].size; g_xc_bytes -= g_xc[best].size; g_xc[best] = g_xc[--g_nxc]; g_xc_reused++; atomic_fetch_add(&g_h0_xr_kept, 1); pthread_mutex_unlock(&g_xcmu); return b; }
    pthread_mutex_unlock(&g_xcmu);
    VkBuffer b = buf_new(want, ALL_BUF_USAGE, m, (void **)p); *size = want; g_xr_fresh++; g_xr_fresh_bytes += want; atomic_fetch_add(&g_h0_xr_fresh, 1); atomic_fetch_add(&g_h0_xr_fresh_b, want);
    pthread_mutex_lock(&g_xcmu); if (g_nxmap < 256) g_xmap[g_nxmap++] = (typeof(g_xmap[0])){*m, *p}; pthread_mutex_unlock(&g_xcmu);
    return b;
}
static void xring_put(VkBuffer b, VkDeviceMemory m, VkDeviceSize size) {   // a freed extra ring: kept for reuse, or destroyed
    pthread_mutex_lock(&g_xcmu); void *p = NULL; int k = 0;
    for (; k < g_nxmap; k++) if (g_xmap[k].m == m) { p = g_xmap[k].p; break; }
    if (p && g_nxc < (ring_reuse_on() ? NXC : 8) && g_xc_bytes + size <= xring_keep() && !(ring_reuse_on() ? g_mem_real_low : g_mem_tight)) { g_xc[g_nxc++] = (typeof(g_xc[0])){b, m, p, size}; g_xc_bytes += size; pthread_mutex_unlock(&g_xcmu); return; }
    atomic_fetch_add(&g_h0_xr_nokeep[!p ? 0 : g_nxc >= (ring_reuse_on() ? NXC : 8) ? 1 : g_xc_bytes + size > xring_keep() ? 2 : 3], 1);   // H0: why it was not kept
    if (k < g_nxmap) g_xmap[k] = g_xmap[--g_nxmap];
    pthread_mutex_unlock(&g_xcmu);
    vkDestroyBuffer(g_dev, b, NULL); vkFreeMemory(g_dev, m, NULL);
}
static VkDeviceSize ring_alloc(VkDeviceSize n, VkDeviceSize align) {
    VkDeviceSize o = (C->used + align - 1) / align * align;
    if (o + n > C->ring_size) {   // full: an extra ring for the rest of this submission (what was recorded keeps its buffer), freed when the submission is done
        if (n > RING_XMAX) { RED_ONCE(1, "one copy of %llu MB is larger than a ring buffer: dropped", (unsigned long long)(n >> 20)); return VK_WHOLE_SIZE; }
        VkDeviceSize xs = C->ring0_size > RING_MIN ? C->ring0_size : 16u << 20; while (xs < n) xs *= 2; if (xs > RING_XMAX) xs = RING_XMAX;
        if (ring_reuse_on() && !C->nextra && g_h05_over) {   // DM_VK_RING_REUSE: the first extra ring takes what the recent submissions overflowed by
            VkDeviceSize w = g_h05_over + g_h05_over / 4; if (w < n) w = n; w = (w + (32u << 20) - 1) & ~(VkDeviceSize)((32u << 20) - 1); if (w > RING_XMAX) w = RING_XMAX;
            if (w > xs) { xs = w; g_h05_sized++; } }
        C->extra = realloc(C->extra, (C->nextra + 1) * sizeof *C->extra);
        C->extra[C->nextra].b = xring_get(xs, &xs, &C->extra[C->nextra].m, &C->ringp); C->extra[C->nextra].size = xs; C->ring = C->extra[C->nextra++].b;
        { uint64_t now_x = atomic_fetch_add(&g_extra_ring_bytes, xs) + xs, pk = atomic_load(&g_xr_peak); while (now_x > pk && !atomic_compare_exchange_weak(&g_xr_peak, &pk, now_x)) {} }
        atomic_fetch_add(&g_xring_n, 1); C->ring_size = xs;
        C->used = 0; o = 0; g_wgen++; g_nwlog = 0; g_h0_gen[1]++;   // copies cached in the old ring are not reused from the new one
    }
    C->total += o + n - C->used; C->used = o + n; return o;
}
static VkDeviceSize ring_put(const void *src, VkDeviceSize n, VkDeviceSize align) {
    VkDeviceSize o = ring_alloc(n, align);
    if (o != VK_WHOLE_SIZE) memcpy(C->ringp + o, src, n);
    return o;
}
static void post_add(post_t p) { if (C->npost == C->cpost) { C->cpost = C->cpost ? 2 * C->cpost : 256; C->post = realloc(C->post, C->cpost * sizeof *C->post); } C->post[C->npost++] = p; }
static uint64_t gpu_ticks(void) { uint64_t g = guest_ns(); if (!g) g = now_ns(); return g * 12 / 625; }   // the GPU's 19.2 MHz timer, as nvn_mtl.m reports it
// ---------- What the GPU writes: storage buffers of compute dispatches ----------
// Draws and dispatches read guest memory through copies made when they are recorded, so what a dispatch writes into its storage buffers (particles simulated on the GPU,
// the exposure an auto-exposure pass computes) lives only in the ring until its submission completes and the copy goes back to guest memory (post kind 3). Until then
// later work recorded from the same range is bound to that ring range itself instead of a fresh copy of the guest's stale bytes (gw_find); the entry dies with the ring
// (ctx_complete, when the bytes are back in guest memory)
typedef struct { uint64_t addr, size; VkBuffer buf; VkDeviceSize off; uint8_t *p; uint64_t fence; int ci;   // ci: the context whose ring holds it
                 const uint8_t *snap; int exact; uint64_t chk; int dirty; uint8_t dfr;
                 VkBuffer fbuf; VkDeviceSize foff; uint8_t *fp; uint64_t ffence; int fci; uint8_t pad; } gw_t;   // pad: a padded window of small copies (DM_VK_GW_PAD)   // f*: its forwarded copy (DM_VK_GW_FORWARD): in the ring of submission ffence, context fci   // dfr: written by a dispatch deferred past the open pass (defer_flush), not recorded yet
// snap: the guest's bytes the dispatch was recorded with (its post kind 5 owns them; cleared here before they are freed), to tell which words the CPU rewrote since (gw_cpu_dirty);
// exact: an exact write (a texture-to-buffer or buffer copy writes every byte of its range), not a dispatch's whole bound storage buffer; chk / dirty: gw_cpu_dirty's answer for a submission
enum { NGW = 512 };
static gw_t g_gw[NGW]; static int g_ngw; static pthread_mutex_t g_gwmu = PTHREAD_MUTEX_INITIALIZER;
static void xr_raise(VkBuffer b, uint64_t until);   // an extra ring a later submission reads stays until it is done (see g_xr)
// Which megabytes of guest memory any entry covers (a count per megabyte, folded into 64K slots): a read touching none of them has no entry to find, and gw_find / gw_exact / gw_any
// answer it without the lock and the walk over every entry -- every uniform, storage and texel buffer of every draw asks, and the entries in flight grow with the GPU's lag.
// Changed under g_gwmu; read without it on the submit thread, which alone adds entries (a removal seen late only sends the read to the walk)
enum { GWPG = 1 << 16 };
static uint16_t g_gw_pg[GWPG];
static uint64_t g_gw_calls, g_gw_walked, g_gw_quick;   // gw_find calls, entries looked at, calls answered by the megabyte counts (the recording line)
static void gw_pg_mark(uint64_t addr, uint64_t size, int d) {
    if (!size) return;
    uint64_t a = addr >> 20, b = (addr + size - 1) >> 20; if (b - a >= GWPG) b = a + GWPG - 1;
    for (uint64_t m = a; m <= b; m++) __atomic_add_fetch(&g_gw_pg[m & (GWPG - 1)], (uint16_t)d, __ATOMIC_RELAXED);
}
static int gw_pg_any(uint64_t addr, uint64_t size) {
    if (!size) return 0;
    uint64_t a = addr >> 20, b = (addr + size - 1) >> 20; if (b - a >= GWPG) return 1;
    for (uint64_t m = a; m <= b; m++) if (__atomic_load_n(&g_gw_pg[m & (GWPG - 1)], __ATOMIC_RELAXED)) return 1;
    return 0;
}
// DM_VK_GW_INDEX (Harvest Moon heavy scene; off by default): the walks over the GPU-written entries (gw_find, gw_any, gw_compose) test each entry's range in two compact arrays
// beside g_gw (16 bytes an entry) instead of in the ~130-byte entries: hm9 walked 270 entries a lookup, 875 lookups a frame, with the lock held 10 ms a frame by the recording
// thread. The arrays are kept in step with g_gw always (gw_add / gw_del / the oldest dropped); the switch only picks which the walks read. Same answers by construction
static uint64_t g_gwa[NGW], g_gwe[NGW];   // [i]: g_gw[i].addr, g_gw[i].addr + g_gw[i].size
static int gw_index_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_GW_INDEX"); v = e && atoi(e) != 0; if (v) LOG("vk: DM_VK_GW_INDEX=1: GPU-written entries walked by their compact ranges"); } return v; }
#define GW_MISS(i, lo, n) (gw_index_on() ? ((lo) >= g_gwe[i] || (lo) + (n) <= g_gwa[i]) : ((lo) >= g_gw[i].addr + g_gw[i].size || (lo) + (n) <= g_gw[i].addr))   // entry i does not overlap [lo, lo + n)
static void gw_del(int i) { gw_pg_mark(g_gw[i].addr, g_gw[i].size, -1); g_gw[i] = g_gw[--g_ngw]; g_gwa[i] = g_gwa[g_ngw]; g_gwe[i] = g_gwe[g_ngw]; }   // caller holds g_gwmu
static int g_gw_add_dfr;   // gw_add: the entry is a deferred dispatch's (see defer_on)
static int g_gw_add_pad;   // gw_add: the entry is a padded window (DM_VK_GW_PAD)
static int g_defer_haz;   // a draw's resources include what a deferred dispatch writes: the pass ends and the dispatches are recorded before the draw (DRAWQ)
static void gw_add(uint64_t addr, uint64_t size, VkBuffer buf, VkDeviceSize off, uint8_t *p, uint64_t fence, int ci, const uint8_t *snap, int exact) {
    lk_(&g_gwmu, LW_GWMU);
    for (int i = 0; i < g_ngw; i++) if (g_gw[i].addr == addr && g_gw[i].size == size) { gw_del(i); break; }   // the newer write of one range replaces the older
    if (g_ngw == NGW) { RED_ONCE(62, "more than %d GPU-written buffer ranges in flight: the oldest is forgotten", NGW); gw_pg_mark(g_gw[0].addr, g_gw[0].size, -1); memmove(g_gw, g_gw + 1, (NGW - 1) * sizeof *g_gw); memmove(g_gwa, g_gwa + 1, (NGW - 1) * 8); memmove(g_gwe, g_gwe + 1, (NGW - 1) * 8); g_ngw--; }
    g_gwa[g_ngw] = addr; g_gwe[g_ngw] = addr + size;
    g_gw[g_ngw++] = (gw_t){addr, size, buf, off, p, fence, ci, snap, exact, 0, 0, (uint8_t)g_gw_add_dfr}; g_gw[g_ngw - 1].pad = (uint8_t)g_gw_add_pad; gw_pg_mark(addr, size, 1);
    pthread_mutex_unlock(&g_gwmu);
}
// The ring range holding what the GPU wrote at [addr, addr + size), when an entry covers it and the offset in it keeps `align`. An entry of the submission being recorded is
// used as it is. One of an earlier submission still in flight is used too (may_hold): the submissions run in order on the GPU, so this one reads what that one wrote, and that
// one's ring is held from reuse until this one is done (ctx hold). That used to be a wait for the earlier submission -- the GPU drained in the middle of the frame, every frame,
// since Unreal updates its GPU scene and culls with compute and draws from the results in the next command buffer. Without may_hold (indirect arguments the CPU reads) the earlier
// submission is waited for and its bytes read from guest memory, where the completion put them: 0 then, as for no entry. g_gw_ci: the context of the ring of the last hit
int mtl_wait(uint64_t fence, uint64_t timeout_ns);
static int g_gw_ci; static _Atomic uint64_t g_gw_wait_ns, g_gw_waits, g_gw_held, g_gw_copied, g_gw_why[3], g_gw_b2t, g_gw_b2b, g_gw_gpu_reads, g_gw_cpu_fix, g_gw_b2t_guest;   // b2t / b2b: texture and buffer copies made from GPU-written bytes   // why: 0 indirect draw arguments, 1 range not inside the written one, 2 this submission's, not fitting
static VkDeviceSize ring_alloc(VkDeviceSize n, VkDeviceSize align);
static VkDeviceSize ring_put(const void *src, VkDeviceSize n, VkDeviceSize align);
// Whether the CPU rewrote any of [lo, hi) of a dispatch's range since the dispatch was recorded (the guest's bytes against its snapshot; only the bytes read). On the Switch the GPU
// writes the pool's memory itself, so a later read sees the GPU's bytes where it wrote and the CPU's everywhere else; here the whole bound range was taken from the GPU's ring,
// and bytes the CPU had rewritten there in between (Unreal's per-frame uploads into buffers a compute pass also binds) were read stale. Caller holds g_gwmu
// Cost: a partial read takes the whole range's answer first (one compare a submission): a range the CPU left alone is clean in every part of it. DM_VK_STALE_PATCH=0 leaves
// the comparison out (the GPU's bytes as they are, as before vk19.3). g_t_stale / g_stale_bytes: time and bytes in these compares and patches (the recording line)
static uint64_t g_t_stale, g_stale_bytes, g_t_patch, g_patch_regions, g_npatch;   // g_t_patch: the part in gw_patch; regions: copy regions it recorded
static int stale_patch_mode(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_STALE_PATCH"); v = e ? atoi(e) : 2; if (v < 0 || v > 2) v = 2;   // 2 by default from vk19.44: checked exact on the GPU (Run I, DM_VK_PATCH_VERIFY)
    if (!v) LOG("DM_VK_STALE_PATCH=0: reads of an earlier submission's GPU-written range take its bytes as they are"); if (v == 2) LOG("DM_VK_STALE_PATCH=2: the CPU's rewrites over GPU-written ranges are put in place by a compute dispatch"); } return v; }
static int stale_patch_on(void) { return stale_patch_mode() != 0; }
// DM_VK_GW_CHAIN_SNAP (Harvest Moon's heavy scene, H0b; off by default): a dispatch whose storage buffer was bound from a running submission's GPU-written range (chained: the
// GPU starts from that range's bytes, in its ring) had no snapshot. Its entry then could not tell the CPU's rewrites (gw_cpu_dirty answers 0 with no snapshot), and its completion
// wrote the whole range back, over what the CPU had written meanwhile. With one submission a frame and the GPU behind, every frame's dispatch chains on the last one, and the
// CPU's per-frame writes to such a buffer never reached the GPU nor stayed in guest memory (hm7: 96-206 KB a frame overwritten; the work the game asked the GPU for grew every
// frame). =1: a chained range gets a snapshot as well, of the guest's bytes when the dispatch is recorded; it goes back by changed words (post kind 5), and later reads patch in
// the CPU's rewrites (gw_cpu_dirty / gw_patch). Each write-back also puts the words it wrote into the snapshots of the later live entries over the same bytes, so a snapshot keeps
// meaning "guest memory without the CPU's writes since": an earlier submission's write-back is not taken for a CPU rewrite. =2: as 1, plus counts at each chained write-back of
// the CPU's words kept (which the whole write-back overwrote before) and of words both the CPU and the GPU wrote (the GPU's kept, as before). Per-600 line 'vk: chained GPU-written ranges'
static int chain_snap_mode(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_GW_CHAIN_SNAP"); v = e ? atoi(e) : 0; if (v < 0 || v > 2) v = 1;
    if (v) LOG("vk: DM_VK_GW_CHAIN_SNAP=%d: a dispatch's range bound from a running submission's GPU-written range gets a snapshot and goes back by changed words%s", v, v == 2 ? " (with counts of the CPU's words kept)" : ""); } return v; }
static uint64_t g_cs_n, g_cs_b, g_cs_b2b;   // chained ranges given a snapshot (recording thread); GPU-written buffer copies whose CPU-side bytes went into live snapshots
static _Atomic uint64_t g_cs_prop_b, g_cs_prop_n, g_cs_kept_w, g_cs_both_w, g_cs_wb_n;   // completion: bytes put into later snapshots (writes), CPU words kept, words both wrote, chained write-backs
// Host write-backs into guest memory (post kinds 3, 5, 6 at completion), logged with a sequence number: DM_VK_IDXPOOL_ONCE drops a pool copy one of them landed in since it was
// made (hm9's verify differences: the completion thread writes back while the next submission is recorded)
enum { NHWL = 1024 };
static struct { uint64_t lo, hi; } g_hwl[NHWL]; static uint64_t g_hwl_seq; static pthread_mutex_t g_hwl_mu = PTHREAD_MUTEX_INITIALIZER;
static int idxpool_once_mode(void);
static void hwl_add(const void *dst, size_t n) { if (!idxpool_once_mode()) return; pthread_mutex_lock(&g_hwl_mu); g_hwl[g_hwl_seq % NHWL].lo = (uint64_t)(uintptr_t)dst; g_hwl[g_hwl_seq % NHWL].hi = (uint64_t)(uintptr_t)dst + n; g_hwl_seq++; pthread_mutex_unlock(&g_hwl_mu); }
static int hwl_since(uint64_t seq, uint64_t lo, uint64_t hi) {   // 1: a write-back over [lo, hi) since seq (or the log went round)
    int r = 0; pthread_mutex_lock(&g_hwl_mu);
    if (g_hwl_seq - seq > NHWL) r = 1; else for (uint64_t k = seq; k < g_hwl_seq && !r; k++) r = g_hwl[k % NHWL].lo < hi && g_hwl[k % NHWL].hi > lo;
    pthread_mutex_unlock(&g_hwl_mu); return r;
}
static uint64_t hwl_now(void) { pthread_mutex_lock(&g_hwl_mu); uint64_t v = g_hwl_seq; pthread_mutex_unlock(&g_hwl_mu); return v; }
// A write-back of n bytes at dst (now: the GPU's final bytes; old: what it started from, NULL for a whole write-back) of submission fence: the bytes it changes in guest memory go into
// the snapshots of the later live entries over them. Caller holds g_gwmu
static void gw_snap_follow(const uint8_t *dst, size_t n, const uint8_t *now, const uint8_t *old, uint64_t fence) {
    uint64_t a = (uint64_t)(uintptr_t)dst, b = a + n;
    for (int k = 0; k < g_ngw; k++) { gw_t *e = &g_gw[k];
        if (!e->snap || e->fence <= fence || e->addr >= b || e->addr + e->size <= a) continue;
        uint64_t lo = e->addr > a ? e->addr : a, hi = e->addr + e->size < b ? e->addr + e->size : b, cnt = 0;
        uint8_t *es = (uint8_t *)(uintptr_t)e->snap;   // the snapshot buffers are the backend's own (snap_alloc)
        for (uint64_t x = lo; x < hi; x++) { uint8_t v = now[x - a]; if ((!old || v != old[x - a]) && es[x - e->addr] != v) { es[x - e->addr] = v; cnt++; } }
        if (cnt) { e->chk = 0; atomic_fetch_add(&g_cs_prop_b, cnt); atomic_fetch_add(&g_cs_prop_n, 1); }
    }
}
static int gw_whole_dirty(gw_t *e) {
    if (e->chk != g_fence + 1) { e->chk = g_fence + 1; e->dirty = memcmp((const void *)(uintptr_t)e->addr, e->snap, e->size) != 0; g_stale_bytes += e->size; }
    return e->dirty;
}
static int gw_cpu_dirty(gw_t *e, uint64_t lo, uint64_t hi) {
    if (!e->snap || hi <= lo || !stale_patch_on()) return 0;
    uint64_t t0 = stat_ns(); int r = gw_whole_dirty(e);   // the whole range: once a submission
    if (r && !(lo == e->addr && hi == e->addr + e->size)) { r = memcmp((const void *)(uintptr_t)lo, e->snap + (lo - e->addr), hi - lo) != 0; g_stale_bytes += hi - lo; }
    g_t_stale += stat_ns() - t0;
    return r;
}
// The words of [lo, hi) the CPU rewrote since e was recorded, copied from the guest over dst (where lo lands), after the GPU's bytes were copied there (the upload command buffer).
// Compared 8 bytes at a time (64 at a time over equal stretches); only the bytes from the first rewritten word to the last go into the ring
static inline uint64_t ld64(const uint8_t *p) { uint64_t v; memcpy(&v, p, 8); return v; }
static inline uint32_t ld32(const uint8_t *p) { uint32_t v; memcpy(&v, p, 4); return v; }
// DM_VK_STALE_PATCH=2 (vk19.39): the runs go to the GPU as a table and one compute dispatch copies them (a thread a run, at most 64 words each), where mode 1 records a
// copy region for each: ~600-1,300 regions a frame in the dungeon cost the upload command buffers 4.6 ms of GPU time (Run B: 0.2 ms without any patch). The shader, compiled
// with the NDK's glslc and kept here as words (our own source, below):
//   layout(local_size_x = 64) in; readonly buffer Src { uint s[]; } (binding 0); buffer Dst { uint d[]; } (1); readonly buffer Tab { uvec4 t[]; } (2); push constant uint count;
//   main: i = gl_GlobalInvocationID.x; if (i >= count) return; uvec4 e = t[i]; for (k < e.z) d[e.x + k] = s[e.y + k];
// Runs not on 4-byte boundaries, or a range past maxStorageBufferRange, keep the copy regions
static const uint32_t k_patch_spv[] = {
0x07230203, 0x00010300, 0x000d000a, 0x00000058, 0x00000000, 0x00020011, 0x00000001, 0x0006000b, 0x00000001, 0x4c534c47,
0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x0006000f, 0x00000005, 0x00000004, 0x6e69616d,
0x00000000, 0x0000000b, 0x00060010, 0x00000004, 0x00000011, 0x00000040, 0x00000001, 0x00000001, 0x00040047, 0x0000000b,
0x0000000b, 0x0000001c, 0x00050048, 0x00000011, 0x00000000, 0x00000023, 0x00000000, 0x00030047, 0x00000011, 0x00000002,
0x00040047, 0x00000021, 0x00000006, 0x00000010, 0x00040048, 0x00000022, 0x00000000, 0x00000018, 0x00050048, 0x00000022,
0x00000000, 0x00000023, 0x00000000, 0x00030047, 0x00000022, 0x00000002, 0x00040047, 0x00000024, 0x00000022, 0x00000000,
0x00040047, 0x00000024, 0x00000021, 0x00000002, 0x00040047, 0x00000034, 0x00000006, 0x00000004, 0x00050048, 0x00000035,
0x00000000, 0x00000023, 0x00000000, 0x00030047, 0x00000035, 0x00000002, 0x00040047, 0x00000037, 0x00000022, 0x00000000,
0x00040047, 0x00000037, 0x00000021, 0x00000001, 0x00040047, 0x0000003c, 0x00000006, 0x00000004, 0x00040048, 0x0000003d,
0x00000000, 0x00000018, 0x00050048, 0x0000003d, 0x00000000, 0x00000023, 0x00000000, 0x00030047, 0x0000003d, 0x00000002,
0x00040047, 0x0000003f, 0x00000022, 0x00000000, 0x00040047, 0x0000003f, 0x00000021, 0x00000000, 0x00040047, 0x0000004d,
0x0000000b, 0x00000019, 0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00040015, 0x00000006, 0x00000020,
0x00000000, 0x00040017, 0x00000009, 0x00000006, 0x00000003, 0x00040020, 0x0000000a, 0x00000001, 0x00000009, 0x0004003b,
0x0000000a, 0x0000000b, 0x00000001, 0x0004002b, 0x00000006, 0x0000000c, 0x00000000, 0x00040020, 0x0000000d, 0x00000001,
0x00000006, 0x0003001e, 0x00000011, 0x00000006, 0x00040020, 0x00000012, 0x00000009, 0x00000011, 0x0004003b, 0x00000012,
0x00000013, 0x00000009, 0x00040015, 0x00000014, 0x00000020, 0x00000001, 0x0004002b, 0x00000014, 0x00000015, 0x00000000,
0x00040020, 0x00000016, 0x00000009, 0x00000006, 0x00020014, 0x00000019, 0x00040017, 0x0000001e, 0x00000006, 0x00000004,
0x0003001d, 0x00000021, 0x0000001e, 0x0003001e, 0x00000022, 0x00000021, 0x00040020, 0x00000023, 0x0000000c, 0x00000022,
0x0004003b, 0x00000023, 0x00000024, 0x0000000c, 0x00040020, 0x00000026, 0x0000000c, 0x0000001e, 0x0003001d, 0x00000034,
0x00000006, 0x0003001e, 0x00000035, 0x00000034, 0x00040020, 0x00000036, 0x0000000c, 0x00000035, 0x0004003b, 0x00000036,
0x00000037, 0x0000000c, 0x0003001d, 0x0000003c, 0x00000006, 0x0003001e, 0x0000003d, 0x0000003c, 0x00040020, 0x0000003e,
0x0000000c, 0x0000003d, 0x0004003b, 0x0000003e, 0x0000003f, 0x0000000c, 0x0004002b, 0x00000006, 0x00000040, 0x00000001,
0x00040020, 0x00000045, 0x0000000c, 0x00000006, 0x0004002b, 0x00000014, 0x0000004a, 0x00000001, 0x0004002b, 0x00000006,
0x0000004c, 0x00000040, 0x0006002c, 0x00000009, 0x0000004d, 0x0000004c, 0x00000040, 0x00000040, 0x00050036, 0x00000002,
0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x000300f7, 0x0000004e, 0x00000000, 0x000300fb, 0x0000000c,
0x0000004f, 0x000200f8, 0x0000004f, 0x00050041, 0x0000000d, 0x0000000e, 0x0000000b, 0x0000000c, 0x0004003d, 0x00000006,
0x0000000f, 0x0000000e, 0x00050041, 0x00000016, 0x00000017, 0x00000013, 0x00000015, 0x0004003d, 0x00000006, 0x00000018,
0x00000017, 0x000500ae, 0x00000019, 0x0000001a, 0x0000000f, 0x00000018, 0x000300f7, 0x0000001c, 0x00000000, 0x000400fa,
0x0000001a, 0x0000001b, 0x0000001c, 0x000200f8, 0x0000001b, 0x000200f9, 0x0000004e, 0x000200f8, 0x0000001c, 0x00060041,
0x00000026, 0x00000027, 0x00000024, 0x00000015, 0x0000000f, 0x0004003d, 0x0000001e, 0x00000028, 0x00000027, 0x000200f9,
0x0000002a, 0x000200f8, 0x0000002a, 0x000700f5, 0x00000006, 0x00000057, 0x0000000c, 0x0000001c, 0x0000004b, 0x0000002b,
0x00050051, 0x00000006, 0x00000032, 0x00000028, 0x00000002, 0x000500b0, 0x00000019, 0x00000033, 0x00000057, 0x00000032,
0x000400f6, 0x0000002c, 0x0000002b, 0x00000000, 0x000400fa, 0x00000033, 0x0000002b, 0x0000002c, 0x000200f8, 0x0000002b,
0x00050051, 0x00000006, 0x00000039, 0x00000028, 0x00000000, 0x00050080, 0x00000006, 0x0000003b, 0x00000039, 0x00000057,
0x00050051, 0x00000006, 0x00000042, 0x00000028, 0x00000001, 0x00050080, 0x00000006, 0x00000044, 0x00000042, 0x00000057,
0x00060041, 0x00000045, 0x00000046, 0x0000003f, 0x00000015, 0x00000044, 0x0004003d, 0x00000006, 0x00000047, 0x00000046,
0x00060041, 0x00000045, 0x00000048, 0x00000037, 0x00000015, 0x0000003b, 0x0003003e, 0x00000048, 0x00000047, 0x00050080,
0x00000006, 0x0000004b, 0x00000057, 0x0000004a, 0x000200f9, 0x0000002a, 0x000200f8, 0x0000002c, 0x000200f9, 0x0000004e,
0x000200f8, 0x0000004e, 0x000100fd, 0x00010038
};
static VkPipeline g_patch_pipe; static VkPipelineLayout g_patch_pl; static VkDescriptorSetLayout g_patch_dsl; static int g_patch_ok = -1;
static uint64_t g_patch_disp, g_patch_runs;
static int patch_pipe(void) {
    if (g_patch_ok >= 0) return g_patch_ok;
    g_patch_ok = 0;
    VkDescriptorSetLayoutBinding b[3]; for (int i = 0; i < 3; i++) b[i] = (VkDescriptorSetLayoutBinding){(uint32_t)i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL};
    VkDescriptorSetLayoutCreateInfo li = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 3, b};
    if (vkCreateDescriptorSetLayout(g_dev, &li, NULL, &g_patch_dsl) != VK_SUCCESS) return 0;
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 4};
    VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &g_patch_dsl, 1, &pr};
    if (vkCreatePipelineLayout(g_dev, &pli, NULL, &g_patch_pl) != VK_SUCCESS) return 0;
    VkShaderModuleCreateInfo mi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sizeof k_patch_spv, k_patch_spv}; VkShaderModule m;
    if (vkCreateShaderModule(g_dev, &mi, NULL, &m) != VK_SUCCESS) return 0;
    VkComputePipelineCreateInfo ci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0, {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, m, "main", NULL}, g_patch_pl, VK_NULL_HANDLE, 0};
    VkResult r = vkCreateComputePipelines(g_dev, VK_NULL_HANDLE, 1, &ci, NULL, &g_patch_pipe); vkDestroyShaderModule(g_dev, m, NULL);
    if (r != VK_SUCCESS) { LOG("[RED] vk: the stale-patch compute pipeline was not made (%d): copy regions instead", (int)r); return 0; }
    g_patch_ok = 1; return 1;
}
static VkDescriptorSet patch_set(void);
// DM_VK_PATCH_VERIFY=1 (diagnostics, with DM_VK_STALE_PATCH=2): each patch dispatch is checked when its submission completes, every word of its table against the staged
// words (the ring is host-visible). A mismatch is either a wrong patch or a later write of this submission over the same bytes (a dispatch storing there): both are logged
enum { NPV = 64 };
static struct { const uint32_t *dst, *src, *tab, *snap; uint32_t n; } g_pv[32][NPV]; static int g_npv[32]; static uint64_t g_pv_words, g_pv_bad, g_pv_checked, g_pv_snap_words, g_pv_snap_bad;
static int patch_verify(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_PATCH_VERIFY"); v = e && atoi(e) != 0; } return v; }
// Two checks: snap, a copy of the destination taken on the GPU right after the dispatch (the patch itself: should be 0 differences), and dst, the destination when the
// submission is done (later work of the submission, a dispatch storing into the same buffer, may have changed it since: differences there alone are not a fault)
static void patch_verify_ctx(int ci) {
    for (int i = 0; i < g_npv[ci]; i++) { const uint32_t *t = g_pv[ci][i].tab, *sn = g_pv[ci][i].snap;
        if (sn) for (uint32_t j = 0; j < g_pv[ci][i].n; j++) { const uint32_t *d = sn + t[4 * j], *sw = g_pv[ci][i].src + t[4 * j + 1];
            for (uint32_t k = 0; k < t[4 * j + 2]; k++) { g_pv_snap_words++; if (d[k] != sw[k]) { g_pv_snap_bad++; static int nl; if (nl++ < 12) LOG("[RED] vk: stale patch check, right after the dispatch: run %u (dst word %u, src word %u, %u words) word %u is %#x, staged %#x", j, t[4 * j], t[4 * j + 1], t[4 * j + 2], k, d[k], sw[k]); } } }
        for (uint32_t j = 0; j < g_pv[ci][i].n; j++) { const uint32_t *d = g_pv[ci][i].dst + t[4 * j], *sw = g_pv[ci][i].src + t[4 * j + 1];
            for (uint32_t k = 0; k < t[4 * j + 2]; k++) { g_pv_words++; if (d[k] != sw[k]) { g_pv_bad++; static int nl; if (nl++ < 4) LOG("vk: stale patch check, at completion: word %u of run %u is %#x, staged %#x (later work may have written it)", k, j, d[k], sw[k]); } } }
        g_pv_checked++; }
    g_npv[ci] = 0;
}
// The runs rg[0..nr) (srcOffset: from lo, size: bytes) as a dispatch: returns 0 when it cannot (the caller records the copy regions)
static int gw_patch_cs(const uint8_t *g, const VkBufferCopy *rg, size_t nr, VkBuffer dst, VkDeviceSize dst_off) {
    if (stale_patch_mode() != 2 || !patch_pipe()) return 0;
    VkDeviceSize sal = g_props.limits.minStorageBufferOffsetAlignment ? g_props.limits.minStorageBufferOffsetAlignment : 256; if (sal < 256) sal = 256;
    uint64_t first = rg[0].srcOffset, last = rg[nr - 1].srcOffset + rg[nr - 1].size;
    if ((dst_off + first) % 4) return 0;
    size_t nt = 0; for (size_t k = 0; k < nr; k++) { if (rg[k].srcOffset % 4 || rg[k].size % 4) return 0; nt += (rg[k].size / 4 + 63) / 64; }
    VkDeviceSize dbase = (dst_off + first) / sal * sal, dlen = dst_off + last - dbase, slen = last - first;
    if (dlen > g_props.limits.maxStorageBufferRange || slen > g_props.limits.maxStorageBufferRange || nt > (1u << 20)) return 0;
    const uint8_t *dmap = dst == C->ring ? C->ringp + (dbase - 0) : NULL;   // the destination's mapping, for DM_VK_PATCH_VERIFY (the caller's current ring)
    VkDeviceSize so = ring_put(g + first, slen, sal); if (so == VK_WHOLE_SIZE) return 0; VkBuffer sb = C->ring; const uint8_t *smap = C->ringp + so;
    VkDeviceSize to = ring_alloc(nt * 16, sal); if (to == VK_WHOLE_SIZE) return 0; VkBuffer tb = C->ring; uint32_t *tab = (uint32_t *)(C->ringp + to);   // after the source: ring_alloc may move to an extra ring
    size_t j = 0;
    for (size_t k = 0; k < nr; k++) for (uint64_t w = 0; w < rg[k].size / 4; w += 64) {
        uint64_t nw = rg[k].size / 4 - w; if (nw > 64) nw = 64;
        tab[4 * j] = (uint32_t)((dst_off + rg[k].srcOffset - dbase) / 4 + w); tab[4 * j + 1] = (uint32_t)((rg[k].srcOffset - first) / 4 + w); tab[4 * j + 2] = (uint32_t)nw; tab[4 * j + 3] = 0; j++;
    }
    VkDescriptorSet ds = patch_set(); if (!ds) return 0;
    VkDescriptorBufferInfo bi[3] = {{sb, so, slen}, {dst, dbase, (dlen + 3) & ~3ull}, {tb, to, nt * 16}};
    VkWriteDescriptorSet w[3]; for (int i = 0; i < 3; i++) w[i] = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, ds, (uint32_t)i, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &bi[i], NULL};
    vkUpdateDescriptorSets(g_dev, 3, w, 0, NULL);
    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
    vkCmdPipelineBarrier(C->up, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);   // after the GPU's bytes landed
    vkCmdBindPipeline(C->up, VK_PIPELINE_BIND_POINT_COMPUTE, g_patch_pipe);
    vkCmdBindDescriptorSets(C->up, VK_PIPELINE_BIND_POINT_COMPUTE, g_patch_pl, 0, 1, &ds, 0, NULL);
    uint32_t cnt = (uint32_t)nt; vkCmdPushConstants(C->up, g_patch_pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &cnt);
    vkCmdDispatch(C->up, (cnt + 63) / 64, 1, 1);
    VkMemoryBarrier mb2 = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
    vkCmdPipelineBarrier(C->up, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb2, 0, NULL, 0, NULL);   // before a later copy or patch over the same bytes in this command buffer
    C->up_used = 1; g_patch_disp++; g_patch_runs += nt;
    if (patch_verify() && dmap && g_ci < 32 && g_npv[g_ci] < NPV) {
        const uint32_t *snap = NULL; VkDeviceSize vo = ring_alloc((dlen + 3) & ~3ull, 16);
        if (vo != VK_WHOLE_SIZE) { VkBufferCopy vc = {dbase, vo, (dlen + 3) & ~3ull}; vkCmdCopyBuffer(C->up, dst, C->ring, 1, &vc); snap = (const uint32_t *)(C->ringp + vo); }   // after mb2: the dispatch's writes are visible to it
        g_pv[g_ci][g_npv[g_ci]++] = (typeof(g_pv[0][0])){(const uint32_t *)dmap, (const uint32_t *)smap, tab, snap, (uint32_t)nt};
    }
    return 1;
}
static void gw_patch(const gw_t *e, uint64_t lo, uint64_t hi, VkBuffer dst, VkDeviceSize dst_off) {
    if (!e->snap) return;
    uint64_t t0 = now_ns();
    const uint8_t *g = (const uint8_t *)(uintptr_t)lo, *sn = e->snap + (lo - e->addr); uint64_t n = hi - lo;
    static VkBufferCopy *rg; static size_t crg; size_t nr = 0; uint64_t i = 0;   // the runs of rewritten words (exact: the bytes between them may be the GPU's); under g_gwmu
    while (i < n) {
        if (i + 64 <= n && !memcmp(g + i, sn + i, 64)) { i += 64; continue; }
        if (i + 8 <= n && ld64(g + i) == ld64(sn + i)) { i += 8; continue; }
        if (i + 4 <= n ? ld32(g + i) == ld32(sn + i) : !memcmp(g + i, sn + i, n - i)) { i += i + 4 <= n ? 4 : n - i; continue; }
        uint64_t a = i;
        while (i < n) { if (i + 4 <= n) { if (ld32(g + i) == ld32(sn + i)) break; i += 4; continue; } if (!memcmp(g + i, sn + i, n - i)) break; i = n; }   // word by word: an unchanged word may hold the GPU's bytes
        if (nr == crg) { size_t c2 = crg ? 2 * crg : 256; VkBufferCopy *r2 = realloc(rg, c2 * sizeof *rg); if (!r2) break; rg = r2; crg = c2; }
        rg[nr++] = (VkBufferCopy){a, a, i - a};
    }
    g_stale_bytes += n; g_patch_regions += nr; g_npatch++;
    if (nr && gw_patch_cs(g, rg, nr, dst, dst_off)) nr = 0;   // DM_VK_STALE_PATCH=2: done by a dispatch
    if (nr) {
        uint64_t first = rg[0].srcOffset, last = rg[nr - 1].srcOffset + rg[nr - 1].size;
        VkDeviceSize st = ring_put(g + first, last - first, 16);
        if (st != VK_WHOLE_SIZE) {
            VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT};
            vkCmdPipelineBarrier(C->up, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);   // after the GPU's bytes landed
            for (size_t k = 0; k < nr; k++) { rg[k].dstOffset = dst_off + rg[k].srcOffset; rg[k].srcOffset = st + (rg[k].srcOffset - first); }
            for (size_t k = 0; k < nr; k += 4096) vkCmdCopyBuffer(C->up, C->ring, dst, (uint32_t)(nr - k < 4096 ? nr - k : 4096), rg + k);
            C->up_used = 1;
        }
    }
    g_t_stale += now_ns() - t0; g_t_patch += now_ns() - t0;
}
// A read over [addr, addr + size) that earlier submissions still running wrote parts of (Unreal binds its whole GPU-scene buffer; a dispatch updated a few records of it): the guest's
// bytes copied into the ring, then each written part copied over them from its submission's ring at the start of this submission, oldest first (the upload command buffer; those
// submissions are done by then). Each source ring is held until this submission is done. Parts this submission itself writes are not in it (no such case met). Caller holds g_gwmu
static void pass_end_(int line); static int pass_begin(void); static int g_pass_open_(void);
#define pass_end() pass_end_(__LINE__)   // the line that ends a pass is counted per pass signature (gsig_end: the per-pass table names why its passes end)
// DM_VK_GW_SHADOW (Harvest Moon's culled vegetation; off by default). Each of the ~92 cull dispatches a frame writes a 16-byte header and then the visible instances into
// its storage buffer [X, X + n); the draws' vertex shaders read [X + 16, X + n): inside the dispatch's range, but 16 bytes off minStorageBufferOffsetAlignment, so each read was
// composed, with a render-pass break (hm10: 88 a frame). A misaligned read of a dispatch's range is remembered (address, size of the range; offset, size of the read); the next
// time a dispatch writes that range, a GPU copy of the read's part is made right after it (outside any pass, after its barrier) into an aligned ring block, registered as this
// submission's exact entry over the read's range. The reads then fit that newer entry and are bound in place. A later write over the range in the submission is a newer entry
// still (the read composes as before). Per-600 line 'vk: DM_VK_GW_SHADOW'
static int gw_shadow_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_GW_SHADOW"); v = e && atoi(e) != 0; if (v) LOG("vk: DM_VK_GW_SHADOW=1: misaligned reads of a dispatch's range get an aligned copy made right after the dispatch"); } return v; }
enum { NSHL = 1024 };
static struct { uint64_t ea, es, ro, rs; } g_shl[NSHL]; static uint64_t g_shl_learn, g_shl_made;
static void shl_learn(uint64_t ea, uint64_t es, uint64_t ro, uint64_t rs) { unsigned h = (unsigned)((ea >> 4 ^ ea >> 14) % NSHL); if (g_shl[h].ea != ea || g_shl[h].es != es || g_shl[h].ro != ro || g_shl[h].rs != rs) { g_shl[h].ea = ea; g_shl[h].es = es; g_shl[h].ro = ro; g_shl[h].rs = rs; g_shl_learn++; } }
static int shl_find(uint64_t ea, uint64_t es, uint64_t *ro, uint64_t *rs) { unsigned h = (unsigned)((ea >> 4 ^ ea >> 14) % NSHL); if (g_shl[h].ea != ea || g_shl[h].es != es || g_shl[h].ro + g_shl[h].rs > es) return 0; *ro = g_shl[h].ro; *rs = g_shl[h].rs; return 1; }
// H0: a read about to be composed (gw_compose), against the entry that sent it there: why it does not fit, by caller; the first 16 in full. Caller holds g_gwmu
static void h0_compose(uint64_t addr, uint64_t size, VkDeviceSize align, const gw_t *e) {
    int nov = 0, cur = 0;
    for (int i = 0; i < g_ngw; i++) { const gw_t *x = &g_gw[i]; if (addr >= x->addr + x->size || x->addr >= addr + size) continue; nov++; if (x->fence == g_fence + 1) cur = 1; }
    g_h0_cmp[cur][g_h0_caller < H0C_N ? g_h0_caller : H0C_OTHER]++; g_h0_cmp_bytes += size;
    uint64_t end = addr + size, eend = e->addr + e->size; int inside = addr >= e->addr && end <= eend;
    if (nov > 1) g_h0_cmp_why[H0R_MULTI]++;
    if (e->exact) g_h0_cmp_why[H0R_EXACT]++;
    if (addr < e->addr) g_h0_cmp_why[H0R_BEFORE]++;
    if (end > eend) g_h0_cmp_why[end - eend <= 16 ? H0R_PAST16 : end - eend <= 256 ? H0R_PAST256 : H0R_PASTMORE]++;
    if (inside) g_h0_cmp_why[H0R_MISALIGN]++;
    static const char *cn[H0C_N] = {"vertex", "uniform", "storage", "texel", "indirect", "buffer copy", "image copy", "other"};
    if (g_h0_cmp_detail < 16) { g_h0_cmp_detail++;
        LOG("vk: H0 compose %llu: %s read %#llx +%llu (align %llu) over entry %#llx +%llu (%s, %s submission, ring offset %% align %llu); %d entries overlap; past its end by %lld, before its start by %lld",
            (unsigned long long)g_h0_cmp_detail, cn[g_h0_caller < H0C_N ? g_h0_caller : H0C_OTHER], (unsigned long long)addr, (unsigned long long)size, (unsigned long long)align,
            (unsigned long long)e->addr, (unsigned long long)e->size, e->exact ? "an exact copy's" : "a dispatch's", e->fence == g_fence + 1 ? "this" : "an earlier",
            (unsigned long long)((e->off + (addr > e->addr ? addr - e->addr : 0)) % (align ? align : 1)), nov, end > eend ? (long long)(end - eend) : 0LL, addr < e->addr ? (long long)(e->addr - addr) : 0LL); }
}
static VkDeviceSize gw_compose(uint64_t addr, uint64_t size, VkDeviceSize align) {
    VkDeviceSize ro = ring_put((const void *)(uintptr_t)addr, size, align); if (ro == VK_WHOLE_SIZE) return ro;
    int cur = 0;
    for (int i = 0; i < g_ngw; i++) {   // earlier submissions' parts, oldest first: copied at the start of this one (they are complete by then)
        if (GW_MISS(i, addr, size)) continue;
        gw_t *e = &g_gw[i];
        if (e->fence == g_fence + 1) { cur = 1; continue; }
        uint64_t lo = e->addr > addr ? e->addr : addr, hi = e->addr + e->size < addr + size ? e->addr + e->size : addr + size;
        VkBuffer rb = C->ring; VkBufferCopy bc = {e->off + (lo - e->addr), ro + (lo - addr), hi - lo}; vkCmdCopyBuffer(C->up, e->buf, rb, 1, &bc); C->up_used = 1;
        if (gw_cpu_dirty(e, lo, hi)) { gw_patch(e, lo, hi, rb, ro + (lo - addr)); atomic_fetch_add(&g_gw_cpu_fix, 1); }   // the CPU's rewrites since, over the GPU's bytes
        ctx_t *o = &g_ctx[e->ci]; if (o->hold < g_fence + 1) o->hold = g_fence + 1; xr_raise(e->buf, g_fence + 1);
    }
    if (cur) {   // parts this submission's own dispatches wrote: copied here, in order, after them (outside a render pass: the open one is ended and begun again)
        int was_open = g_pass_open_(); if (was_open) { pass_end(); g_h0_cmp_pass++; }
        for (int i = 0; i < g_ngw; i++) {
            if (GW_MISS(i, addr, size)) continue;
            const gw_t *e = &g_gw[i]; if (e->fence != g_fence + 1) continue;
            uint64_t lo = e->addr > addr ? e->addr : addr, hi = e->addr + e->size < addr + size ? e->addr + e->size : addr + size;
            VkBufferCopy bc = {e->off + (lo - e->addr), ro + (lo - addr), hi - lo}; vkCmdCopyBuffer(C->cb, e->buf, C->ring, 1, &bc); g_h0_cmp_cur_bytes += hi - lo;
        }
        { VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT};
          vkCmdPipelineBarrier(C->cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, NULL, 0, NULL); }   // the copies, for what reads the result
        if (was_open) { pass_begin(); g_h0_cpass = 1; }
    }
    return ro;
}
// DM_VK_GW_FORWARD=K (vk19.39, opt-in, 0 off): a read held on an earlier submission's ring (the GPU's bytes used in place) keeps that context from reuse until the reader is
// done, and Unreal reads its GPU-scene ranges for many submissions after the write: the recorder waited ~5.5 ms a frame for "a later one still reading its ring". When the
// writer is K or more submissions back, the whole range (at most 8 MB) is copied into this submission's ring at its start (the upload command buffer, after everything
// before: the writer is done) and later readers hold that copy's context instead; a copy whose context was reused since is made again. The entry itself (and the writer's
// completion, which puts the bytes back in guest memory and ends it) is unchanged. Hold ages and forwarded MB are counted for the per-600 line
static uint64_t g_fwd_n, g_fwd_bytes, g_hold_n, g_hold_age_sum, g_hold_age_max;
static int gw_forward_k(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_GW_FORWARD"); v = e ? atoi(e) : 0; if (v < 0) v = 0; if (v) LOG("DM_VK_GW_FORWARD=%d: GPU-written ranges read %d or more submissions after their write are copied forward", v, v); } return v; }
static int gw_find_(uint64_t addr, uint64_t size, VkDeviceSize align, VkBuffer *buf, VkDeviceSize *off, uint8_t **p, int may_hold) {
    int hit = 0; uint64_t wait = 0; int why = 0;
    if (!__atomic_load_n(&g_ngw, __ATOMIC_RELAXED)) return 0;
    g_gw_calls++; if (!gw_pg_any(addr, size)) { g_gw_quick++; return 0; }
    lk_(&g_gwmu, LW_GWMU); H0_GWH_T0(); g_gw_walked += (uint64_t)g_ngw; g_h0_look++; g_h0_walk += (uint64_t)g_ngw;
    for (int i = g_ngw - 1; i >= 0 && !hit && !wait; i--) {
        if (GW_MISS(i, addr, size)) continue;
        gw_t *e = &g_gw[i];
        if (e->dfr) g_defer_haz = 1;   // what a deferred dispatch writes: it is recorded first (DRAWQ / pass_end)
        int inside = addr >= e->addr && addr + size <= e->addr + e->size, fits = inside && (e->off + (addr - e->addr)) % align == 0;
        if (e->fence != g_fence + 1) {   // an earlier submission's, still running
            if (may_hold && !inside) {   // the read spans more than was written: composed in this submission's ring (gw_compose)
                if (h0_on()) h0_compose(addr, size, align, e);
                VkDeviceSize ro = gw_compose(addr, size, align < 16 ? 16 : align);
                if (ro == VK_WHOLE_SIZE) { wait = e->fence; why = 1; break; }
                *buf = C->ring; *off = ro; if (p) *p = C->ringp + ro; g_gw_ci = g_ci; hit = 1; atomic_fetch_add(&g_gw_copied, 1); break;
            }
            if (!may_hold) { wait = e->fence; why = 0; break; }
            { uint64_t age = g_fence + 1 - e->fence; g_hold_n++; g_hold_age_sum += age; if (age > g_hold_age_max) g_hold_age_max = age;
              int K = gw_forward_k(); if (K >= NCTX) K = NCTX / 2;   // below the context count: a copy older than K is renewed before its context can come round again
              if (K && inside && fits && age >= (uint64_t)K && e->size <= (8u << 20) && !gw_cpu_dirty(e, addr, addr + size)) {
                  if (e->ffence && e->ffence != g_fence + 1 && (g_ctx[e->fci].fence_val != e->ffence || g_fence + 1 - e->ffence >= (uint64_t)K)) e->ffence = 0;   // its context reused since (fence_val is set at commit: this submission's own copy stays), or old again: a new copy
                  if (!e->ffence) {
                      VkDeviceSize fal = g_props.limits.minStorageBufferOffsetAlignment > 256 ? g_props.limits.minStorageBufferOffsetAlignment : 256;
                      VkDeviceSize ro = ring_alloc(e->size, fal);
                      if (ro != VK_WHOLE_SIZE && C->ring == C->ring0) {   // the context's own ring (an extra ring dies with its submission)
                          VkBufferCopy bc = {e->off, ro, e->size}; vkCmdCopyBuffer(C->up, e->buf, C->ring, 1, &bc); C->up_used = 1;
                          e->fbuf = C->ring; e->foff = ro; e->fp = C->ringp + ro; e->ffence = g_fence + 1; e->fci = g_ci; g_fwd_n++; g_fwd_bytes += e->size;
                      }
                  }
                  if (e->ffence && (e->foff + (addr - e->addr)) % align == 0) {
                      ctx_t *f = &g_ctx[e->fci]; if (f->hold < g_fence + 1) f->hold = g_fence + 1; xr_raise(e->fbuf, g_fence + 1);
                      { ctx_t *o0 = &g_ctx[e->ci]; if (e->ffence == g_fence + 1 && o0->hold < g_fence + 1) o0->hold = g_fence + 1; xr_raise(e->buf, g_fence + 1); }   // the copy itself reads the writer's ring in this submission
                      *buf = e->fbuf; *off = e->foff + (addr - e->addr); if (p) *p = e->fp + (addr - e->addr); g_gw_ci = e->fci; hit = 1; atomic_fetch_add(&g_gw_held, 1); break;
                  }
              } }
            ctx_t *o = &g_ctx[e->ci]; if (o->hold < g_fence + 1) o->hold = g_fence + 1; xr_raise(e->buf, g_fence + 1);
            if (inside && gw_cpu_dirty(e, addr, addr + size)) {   // the CPU rewrote words of it since: the GPU's bytes and then the CPU's rewrites over them, in this submission's ring
                VkDeviceSize ro = ring_alloc(size, align < 16 ? 16 : align);
                if (ro == VK_WHOLE_SIZE) { wait = e->fence; why = 1; break; }
                VkBuffer rb = C->ring; uint8_t *rp = C->ringp + ro;
                VkBufferCopy bc = {e->off + (addr - e->addr), ro, size}; vkCmdCopyBuffer(C->up, e->buf, rb, 1, &bc); C->up_used = 1;
                gw_patch(e, addr, addr + size, rb, ro);
                *buf = rb; *off = ro; if (p) *p = rp; g_gw_ci = g_ci; hit = 1; atomic_fetch_add(&g_gw_cpu_fix, 1); break;
            }
            atomic_fetch_add(&g_gw_gpu_reads, 1);
            if (!fits) {   // at an offset the binding cannot take: copied into this submission's ring at the start of the submission (the upload command buffer), when it is complete
                VkDeviceSize ro = ring_alloc(size, align < 16 ? 16 : align);
                if (ro == VK_WHOLE_SIZE) { wait = e->fence; why = 1; break; }
                VkBufferCopy bc = {e->off + (addr - e->addr), ro, size}; vkCmdCopyBuffer(C->up, e->buf, C->ring, 1, &bc); C->up_used = 1;
                *buf = C->ring; *off = ro; if (p) *p = C->ringp + ro; g_gw_ci = g_ci; hit = 1; atomic_fetch_add(&g_gw_copied, 1); break;
            }
            atomic_fetch_add(&g_gw_held, 1);
        } else if (!fits) {   // this submission's own write, not covering the read (or at an offset the binding cannot take): composed (gw_compose)
            if (inside && gw_shadow_on() && !e->exact) shl_learn(e->addr, e->size, addr - e->addr, size);   // DM_VK_GW_SHADOW: next time the dispatch is recorded, this read gets its own aligned copy
            if (may_hold) { if (h0_on()) h0_compose(addr, size, align, e); VkDeviceSize ro = gw_compose(addr, size, align < 16 ? 16 : align);
                            if (ro != VK_WHOLE_SIZE) { *buf = C->ring; *off = ro; if (p) *p = C->ringp + ro; g_gw_ci = g_ci; hit = 1; atomic_fetch_add(&g_gw_copied, 1); break; } }
            why = 2; atomic_fetch_add(&g_gw_why[2], 1); RED_ONCE(63, "a read overlaps a GPU-written buffer range of this submission without fitting in it: read from guest memory, which is stale until the write completes"); break; }
        *buf = e->buf; *off = e->off + (addr - e->addr); if (p) *p = e->p + (addr - e->addr); g_gw_ci = e->ci; hit = 1;
    }
    H0_GWH_END(0); pthread_mutex_unlock(&g_gwmu);
    if (wait) { uint64_t t0 = now_ns(); mtl_wait(wait, UINT64_MAX); atomic_fetch_add(&g_gw_wait_ns, now_ns() - t0); atomic_fetch_add(&g_gw_wait_cum, now_ns() - t0); atomic_fetch_add(&g_gw_waits, 1); atomic_fetch_add(&g_gw_why[why], 1); }
    return hit;
}
static int gw_find(uint64_t addr, uint64_t size, VkDeviceSize align, VkBuffer *buf, VkDeviceSize *off, uint8_t **p) { return gw_find_(addr, size, align, buf, off, p, 1); }
// Whether every GPU-written entry over [addr, addr + size) is an exact write, and there is one: a buffer-to-texture copy takes its source from such ranges only; over a dispatch's
// bound range it reads the guest's bytes (Unreal's staging written by the CPU), as on the Switch, where a copy reads the pool's memory
static int gw_exact(uint64_t addr, uint64_t size) {
    if (!__atomic_load_n(&g_ngw, __ATOMIC_RELAXED) || !gw_pg_any(addr, size)) return 0;
    int any = 0, ok = 1; lk_(&g_gwmu, LW_GWMU);
    for (int i = 0; i < g_ngw; i++) { const gw_t *e = &g_gw[i]; if (addr >= e->addr + e->size || addr + size <= e->addr) continue; any = 1; if (!e->exact) ok = 0; }
    pthread_mutex_unlock(&g_gwmu);
    if (any && !ok) atomic_fetch_add(&g_gw_b2t_guest, 1);
    return any && ok;
}
static void gw_expire(uint64_t fence) {   // completion: the submission's writes are in guest memory now, and its rings go
    lk_(&g_gwmu, LW_GWMU); H0_GWH_T0();
    for (int i = 0; i < g_ngw;) if (g_gw[i].fence <= fence) gw_del(i); else i++;
    H0_GWH_END(1); pthread_mutex_unlock(&g_gwmu);
}
// Extra rings kept past their own submission for later ones that read them (under g_gwmu). A later submission can find a range in one after its context completed --
// a dispatch registers the range it read from there as its own write, still in that ring, and a third submission reads it from that entry -- so `until` follows every
// such reader (xr_raise, where gw_find holds a context for a reader), and a ring any live GPU-written entry still points into is not freed either. The release fence
// used to be taken once, when the ring's own context completed: a later reader's write-back then copied from freed memory (FINAL FANTASY RESONANCE, a Limit Burst:
// SIGSEGV in the completion thread's memcpy, its source in the freed ring)
static struct { VkBuffer b; VkDeviceMemory m; uint64_t until; VkDeviceSize size; } *g_xr; static int g_nxr, g_cxr;
static void xr_raise(VkBuffer b, uint64_t until) { for (int i = 0; i < g_nxr; i++) if (g_xr[i].b == b && g_xr[i].until < until) g_xr[i].until = until; }   // caller holds g_gwmu
static void xr_free(uint64_t done) {
    lk_(&g_gwmu, LW_GWMU); H0_GWH_T0();
    for (int i = 0; i < g_nxr;) {
        int used = g_xr[i].until > done; for (int k = 0; k < g_ngw && !used; k++) used = g_gw[k].buf == g_xr[i].b;
        if (used) { i++; continue; }
        xring_put(g_xr[i].b, g_xr[i].m, g_xr[i].size); atomic_fetch_sub(&g_extra_ring_bytes, g_xr[i].size); g_xr[i] = g_xr[--g_nxr];
    }
    H0_GWH_END(1); pthread_mutex_unlock(&g_gwmu);
}
// GPU time per region signature: what each kind of pass or dispatch costs a frame (registered by the submit thread with its description, timed by the completion thread)
enum { NGSIG = 1024 };
static struct { uint64_t sig; uint64_t ns, n, draws; char d[120]; int eline[4]; uint32_t ecnt[4]; } g_gsig[NGSIG];
static pthread_mutex_t g_gsigmu = PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_cur_sig;   // the open pass's signature (pass_begin), for the line that ends it (gsig_end)
static int gsig_slot(uint64_t sig) { uint32_t i = (uint32_t)(sig ^ sig >> 29) % NGSIG; for (int k = 0; k < NGSIG; k++, i = (i + 1) % NGSIG) if (g_gsig[i].sig == sig || !g_gsig[i].sig) return (int)i; return -1; }   // caller holds g_gsigmu
static void gsig_name(uint64_t sig, const char *d) {   // the submit thread: a description for a signature met the first time
    if (!sig) return; pthread_mutex_lock(&g_gsigmu); int i = gsig_slot(sig);
    if (i >= 0 && !g_gsig[i].sig) { g_gsig[i].sig = sig; snprintf(g_gsig[i].d, sizeof g_gsig[i].d, "%s", d); }
    pthread_mutex_unlock(&g_gsigmu);
}
static void gsig_add(uint64_t sig, uint64_t ns, uint32_t draws) {
    if (!sig) return; pthread_mutex_lock(&g_gsigmu); int i = gsig_slot(sig);
    if (i >= 0 && g_gsig[i].sig == sig) { g_gsig[i].ns += ns; g_gsig[i].n++; g_gsig[i].draws += draws; }
    pthread_mutex_unlock(&g_gsigmu);
}
static uint32_t g_scene_w, g_scene_h;   // the last pass with four or more colour targets (a G-buffer): the resolution the game's dynamic resolution picked (DM_UE_DRS)
static void gsig_end(uint64_t sig, int line) {   // the submit thread: pass `sig` ended at nvn_vk.c:line
    if (!sig) return; pthread_mutex_lock(&g_gsigmu); int i = gsig_slot(sig);
    if (i >= 0 && g_gsig[i].sig == sig) { int k = 0; while (k < 4 && g_gsig[i].ecnt[k] && g_gsig[i].eline[k] != line) k++;
        if (k < 4) { g_gsig[i].eline[k] = line; g_gsig[i].ecnt[k]++; } }
    pthread_mutex_unlock(&g_gsigmu);
}
static void gsig_report(uint64_t frames) {   // the 10 costliest passes and dispatches over the last frames, then the sums start again
    static int idx[NGSIG]; pthread_mutex_lock(&g_gsigmu);
    for (int kind = 0; kind < 2; kind++) {
        int n = 0; uint64_t tot = 0;
        for (int i = 0; i < NGSIG; i++) if (g_gsig[i].sig && g_gsig[i].n && (int)(g_gsig[i].sig & 1) == kind) { idx[n++] = i; tot += g_gsig[i].ns; }
        for (int a = 0; a < n; a++) for (int b = a + 1; b < n; b++) if (g_gsig[idx[b]].ns > g_gsig[idx[a]].ns) { int t = idx[a]; idx[a] = idx[b]; idx[b] = t; }
        if (!kind) LOG("vk: scene resolution (the last G-buffer pass): %ux%u", g_scene_w, g_scene_h);
        LOG("vk: costliest %s over the last %llu frames (%d kinds, %.2f ms a frame in all):", kind ? "dispatches" : "render passes", (unsigned long long)frames, n, tot / 1e6 / frames);
        for (int a = 0; a < n && a < 10; a++) { const typeof(g_gsig[0]) *e = &g_gsig[idx[a]];
            char el[96] = ""; int o = 0; for (int k = 0; k < 4 && e->ecnt[k]; k++) o += snprintf(el + o, sizeof el - o, "%s%d x%.1f", k ? ", " : "; ended at nvn_vk.c:", e->eline[k], (double)e->ecnt[k] / frames);
            LOG("vk:   %.2f ms a frame, %.1f a frame, %.0f draws each: %s%s", e->ns / 1e6 / frames, (double)e->n / frames, e->n ? (double)e->draws / e->n : 0.0, e->d, el); }
    }
    for (int i = 0; i < NGSIG; i++) { g_gsig[i].ns = g_gsig[i].n = g_gsig[i].draws = 0; memset(g_gsig[i].ecnt, 0, sizeof g_gsig[i].ecnt); }
    pthread_mutex_unlock(&g_gsigmu);
}
static _Atomic uint64_t g_gpu_kind_ns[3], g_up_gpu_ns;   // g_up_gpu_ns: the upload command buffers' GPU time (texture uploads, ring copies, GPU-written data composed and patched, the start barrier)   // GPU time inside render passes, dispatches, copies (gpu_mark); the rest of a submission's span is time between them
// Snapshots of dispatch storage ranges (post kind 5: ~3.5 MB a frame in FINAL FANTASY RESONANCE's castle) came from malloc and went back to free: at that size each one was fresh
// mapped pages, ~400 page faults a frame on the submit thread (12,450 minor faults a second, 14% of its time in the kernel). They come from a pool now: blocks by power-of-two size
// (4 KB to 64 MB), kept for reuse up to 32 MB in all; the submit thread takes, the completion thread gives back
enum { SNAP_NC = 15 };   // classes 4 KB << k
static void *g_snap_free[SNAP_NC][64]; static int g_snap_n[SNAP_NC]; static uint64_t g_snap_kept; static pthread_mutex_t g_snapmu = PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_snap_hit, g_snap_miss;
static int snap_class(size_t n) { int k = 0; while (k < SNAP_NC && ((size_t)4096 << k) < n) k++; return k; }
static uint8_t *snap_alloc(size_t n) {
    int k = snap_class(n); if (k >= SNAP_NC) return malloc(n);
    pthread_mutex_lock(&g_snapmu); void *p = g_snap_n[k] ? g_snap_free[k][--g_snap_n[k]] : NULL; if (p) { g_snap_kept -= (size_t)4096 << k; g_snap_hit++; } else g_snap_miss++; pthread_mutex_unlock(&g_snapmu);
    return p ? p : malloc((size_t)4096 << k);
}
static void snap_free(void *p, size_t n) {
    int k = snap_class(n); if (k >= SNAP_NC) { free(p); return; }
    pthread_mutex_lock(&g_snapmu);
    if (g_snap_n[k] < 64 && g_snap_kept + ((size_t)4096 << k) <= (32u << 20)) { g_snap_free[k][g_snap_n[k]++] = p; g_snap_kept += (size_t)4096 << k; p = NULL; }
    pthread_mutex_unlock(&g_snapmu);
    free(p);
}
// Counter timestamps (ReportCounter TIMESTAMP / TIMESTAMP_TOP, post kind 2): every report of a submission used to get one time, its completion's, so a game timing its GPU work with a
// top-of-pipe report at the start and a timestamp at the end of one command buffer read 0 (the NVN timing line's "GPU p50 0.00"; Unreal's dynamic resolution steers on it). Now,
// as the Metal host does by command buffer (nvn_mtl.m rep_ticks), TIMESTAMP_TOP takes the submission's GPU start and the others its GPU end: the end is the completion (the
// guest clock when the completion thread saw the fence), the start that minus the submission's GPU span. DM_VK_GPU_TS=0: one time for all, as before
static int gpu_ts_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_GPU_TS"); v = !(e && !strcmp(e, "0")); if (!v) LOG("DM_VK_GPU_TS=0: counter timestamps all take the completion time"); } return v; }
// DM_VK_EARLY_DONE (vk-t9, default 1): a submission's fence is signalled (g_done, what the game's sync waits and fence queries see) as soon as its completion work that
// the game can see is done -- write-backs into guest memory, GPU-written ranges -- and its command and descriptor pools are reset after that. The pool reset is the Adreno
// driver freeing every pass's command memory (Tomodachi Life's island: 182 passes, ~3.9 ms a reset, 11.6 ms a frame of dm-complete) and it used to come first: each guest
// wait on the GPU (SyncWait twice a frame) saw the end of its submission that much later, and later submissions' ends queued behind it. A context being reset is not taken
// (c->resetting); DM_VK_EARLY_DONE=0: the reset first, as before. g_cl_*: the time from the fence to g_done, and the game's waits (mtl_wait), the command-pool line
static int early_done_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_EARLY_DONE"); v = !(e && atoi(e) == 0); LOG("vk: DM_VK_EARLY_DONE=%d: %s", v, v ? "fences signalled before the pool resets" : "pools reset before the fence is signalled"); } return v; }
static _Atomic uint64_t g_gpu_gap_ns, g_gpu_gap_n;   // GPU time between one game submission's end and the next one's start (timestamps; presents in between count as idle): the completion line
static _Atomic uint64_t g_cl_ns, g_cl_n, g_gw8_ns, g_gw8_n, g_gw8_lag, g_gw8_pend;   // ... and per wait: submissions made after the one waited for, submissions still to end up to it   // fence observed -> g_done; the game's waits on the GPU (mtl_wait): time, count
static void ctx_reset(ctx_t *c);
static void ctx_complete(ctx_t *c) {
    if (c->nh0ip) {   // H0: the GPU-written indirect arguments this submission drew with (vertex or index count, instances)
        static uint64_t prev; uint64_t tot = 0, mx = atomic_load(&g_h0_ii_max);
        static uint64_t nsub; int dump = (nsub++ % 300) == 0;   // every 300th submission with such draws: all of its arguments in full (H0, the dark vegetation)
        for (int i = 0; i < c->nh0ip; i++) { uint32_t a[5] = {0}; int idx = c->h0ix[i] & 1; memcpy(a, c->h0ip[i], idx ? 20 : 16); tot += a[1]; atomic_fetch_add(&g_h0_ii_prims, (uint64_t)a[0] * a[1]); if (a[1] > mx) mx = a[1];
                                             if (c->h0ix[i] & 2) atomic_fetch_add(&g_h0_ii_cp_inst, a[1]);
                                             if (idx ? a[4] : a[3]) atomic_fetch_add(&g_h0_ii_fi, 1); if (idx && a[3]) atomic_fetch_add(&g_h0_ii_bv, 1);
                                             if (dump && i < 24) LOG("vk: H0 indirect arguments of submission %llu, draw %d%s: %s %u, instances %u, first %s %u, %s %d, first instance %u", (unsigned long long)c->fence_val, i,
                                                                     c->h0ix[i] & 2 ? " (in a composed pass)" : "", idx ? "indices" : "vertices", a[0], a[1], idx ? "index" : "vertex", a[2], idx ? "vertex offset" : "-", idx ? (int32_t)a[3] : 0, idx ? a[4] : a[3]); }
        for (int i = 0; i < c->nh0ip; i++) {   // H0 (dark vegetation): did the GPU's arguments reach past what was copied (indices past the index window, vertices past the copied streams)?
            uint32_t a[5] = {0}; int idx = c->h0ix[i] & 1; memcpy(a, c->h0ip[i], idx ? 20 : 16); if (!a[0] || !a[1]) continue;
            int past_i = 0, past_v = 0; uint64_t maxv = 0;
            if (idx && c->h0isz[i]) { uint32_t isz = c->h0isz[i]; uint64_t end = ((uint64_t)a[2] + a[0]) * isz; past_i = end > c->h0iw[i];
                if (c->h0ipp[i]) { uint64_t lim = end < c->h0iw[i] ? end : c->h0iw[i], mi = 0; const uint8_t *ib = c->h0ipp[i];
                    for (uint64_t q = (uint64_t)a[2] * isz; q + isz <= lim; q += isz) { uint32_t v = isz == 2 ? *(const uint16_t *)(ib + q) : isz == 4 ? *(const uint32_t *)(ib + q) : ib[q]; if ((isz == 2 && v == 0xffff) || (isz == 4 && v == 0xffffffffu)) continue; if (v > mi) mi = v; }
                    maxv = mi + (uint64_t)(int64_t)(int32_t)a[3]; past_v = c->h0vmax[i] != UINT32_MAX && maxv >= c->h0vmax[i]; } }
            else if (!idx) { maxv = (uint64_t)a[2] + a[0]; past_v = c->h0vmax[i] != UINT32_MAX && maxv > c->h0vmax[i]; }
            if (past_i) atomic_fetch_add(&g_h0_past_i, 1); if (past_v) atomic_fetch_add(&g_h0_past_v, 1);
            static int nlp; if ((past_i || past_v) && nlp++ < 32)
                LOG("[RED] vk: H0 a GPU-written indirect draw reads past its copies (submission %llu, draw %d): %s %u from %u, vertex offset %d, instances %u; index window %u bytes (%u-byte indices)%s; the highest vertex read %llu, vertices copied %u%s",
                    (unsigned long long)c->fence_val, i, idx ? "indices" : "vertices", a[0], a[2], idx ? (int32_t)a[3] : 0, a[1], c->h0iw[i], c->h0isz[i], past_i ? " PAST THE INDEX WINDOW" : "",
                    (unsigned long long)maxv, c->h0vmax[i], past_v ? " PAST THE VERTEX COPY" : "");
        }
        atomic_fetch_add(&g_h0_ii_draws, (uint64_t)c->nh0ip); atomic_fetch_add(&g_h0_ii_inst, tot); atomic_store(&g_h0_ii_max, mx);
        static int nl; if (prev >= 1000 && tot * 2 < prev && nl++ < 40) LOG("vk: H0 indirect instances fell: submission %llu drew %llu instances in %d GPU-written indirect draws, the one before %llu", (unsigned long long)c->fence_val, (unsigned long long)tot, c->nh0ip, (unsigned long long)prev);
        prev = tot; c->nh0ip = 0;
    }
    if (patch_verify()) { int ci = (int)(c - g_ctx); if (ci >= 0 && ci < 32 && g_npv[ci]) patch_verify_ctx(ci); }
    uint64_t ts[3], t_end = gpu_ticks(), t_top = t_end;
    if (vkGetQueryPoolResults(g_dev, c->qp, 0, 3, sizeof ts, ts, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS && ts[2] >= ts[0]) atomic_fetch_add(&g_up_gpu_ns, (uint64_t)((double)(ts[2] - ts[0]) * g_props.limits.timestampPeriod));
    if (vkGetQueryPoolResults(g_dev, c->qp, 0, 2, sizeof ts, ts, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS && ts[1] > ts[0])
        { uint64_t gns = (uint64_t)((double)(ts[1] - ts[0]) * g_props.limits.timestampPeriod); atomic_fetch_add(&g_gpu_ns, gns); atomic_fetch_add(&g_gpu_cum, gns);
          { static uint64_t prev_end; if (prev_end && ts[0] > prev_end) { uint64_t gap = (uint64_t)((double)(ts[0] - prev_end) * g_props.limits.timestampPeriod);   // the GPU's idle time before this submission
                if (gap < 1000000000ull) { atomic_fetch_add(&g_gpu_gap_ns, gap); atomic_fetch_add(&g_gpu_gap_n, 1); } }
            prev_end = ts[1]; }
          uint64_t dt = gns * 12 / 625; if (gpu_ts_on() && dt < t_end) t_top = t_end - dt; }   // the GPU's 19.2 MHz ticks
    if (c->ntq >= 2) { static uint64_t t[NTQ]; uint32_t n = c->ntq & ~1u;
        if (vkGetQueryPoolResults(g_dev, c->tq, 0, n, n * 8, t, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS)
            for (uint32_t i = 0; i + 1 < n; i += 2) if (t[i + 1] > t[i]) { uint64_t ns = (uint64_t)((double)(t[i + 1] - t[i]) * g_props.limits.timestampPeriod);
                atomic_fetch_add(&g_gpu_kind_ns[c->tkind[i / 2] % 3], ns); gsig_add(c->tsig[i / 2], ns, c->tdraws[i / 2]); } }
    for (int i = 0; i < c->npost; i++) {
        post_t *p = &c->post[i];
        if (p->kind == 1) { volatile uint32_t *v = p->dst; if (p->mode == 1) *v += p->v; else *v = p->v; }
        else if (p->kind == 2) { uint64_t t = p->type == 0x10 ? t_top : t_end, rec[2] = {p->type == 0 || p->type == 0x10 ? t : 0, t}; memcpy((void *)p->addr, rec, sizeof rec); }
        else if (p->kind == 4) {   // a segment of counted draws: added to the queue's samples-passed counter (submission order: one completion thread), written out when addr is a report
            static uint64_t total;
            if (p->n == SIZE_MAX) total += 1u << 20;   // more draws than queries: reported visible
            else for (size_t q = 0; q < p->n; q++) { uint64_t v = 0; if (vkGetQueryPoolResults(g_dev, c->oq, ((const uint32_t *)p->src)[q], 1, 8, &v, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) total += v; }
            if (p->mode) total = 0;   // the segment ended in a ResetCounter
            if (p->addr) { uint64_t rec[2] = {total, gpu_ticks()}; memcpy((void *)p->addr, rec, sizeof rec); }
            free((void *)p->src);
        }
        else if (p->kind == 5) {   // a dispatch's storage buffer: the words the GPU changed (see dispatch)
            const uint8_t *snap = (const uint8_t *)(uintptr_t)p->addr, *now = p->src; uint8_t *dst = p->dst; size_t i = 0, nw = 0;
            if (p->mode == 1) { atomic_fetch_add(&g_cs_wb_n, 1);   // a chained range (DM_VK_GW_CHAIN_SNAP)
                if (chain_snap_mode() == 2) { uint64_t kept = 0, both = 0;
                    for (size_t q = 0; q + 4 <= p->n; q += 4) if (memcmp(dst + q, snap + q, 4)) { if (!memcmp(now + q, snap + q, 4)) kept++; else both++; }
                    atomic_fetch_add(&g_cs_kept_w, kept); atomic_fetch_add(&g_cs_both_w, both); } }
            for (; i + 4 <= p->n; i += 4) if (memcmp(now + i, snap + i, 4)) { memcpy(dst + i, now + i, 4); nw++; }
            for (; i < p->n; i++) if (now[i] != snap[i]) dst[i] = now[i];
            // Its entries end here, not at gw_expire below (vk19.58): guest memory holds the GPU's words and the CPU's rewrites now, so a later read copies it from there.
            // Only the snapshot used to be cleared, and the entry lived on until the rest of the submission was written back: a read in between bound the ring's bytes
            // without the CPU's rewrites (gw_cpu_dirty answers 0 with no snapshot), stale for a frame. gw_del moves the last entry into slot k: k is looked at again
            { uint64_t ended = 0; lk_(&g_gwmu, LW_GWMU); H0_GWH_T0();
              if (chain_snap_mode()) gw_snap_follow(dst, p->n, now, snap, c->fence_val);   // the words written: into later entries' snapshots
              for (int k = 0; k < g_ngw;) if (g_gw[k].snap >= snap && g_gw[k].snap < snap + p->n) { gw_del(k); ended++; } else k++;
              H0_GWH_END(1); pthread_mutex_unlock(&g_gwmu);
              if (ended) { atomic_fetch_add(&g_gw_wb_ended, ended); static int once; if (!once) { once = 1; LOG("vk: GPU-written ranges end at their write-back (vk19.58): %llu ended by the first one, submission %llu", (unsigned long long)ended, (unsigned long long)c->fence_val); } } }
            snap_free((void *)snap, p->n); g_wb_words += nw; g_wb_bytes += p->n; if (nw) hwl_add(dst, p->n);
        }
        else if (p->kind == 6) { hwl_add(p->dst, (size_t)p->v * p->mode); for (uint32_t r = 0; r < p->v; r++) memcpy((uint8_t *)p->dst + (size_t)r * p->mode, p->src + (size_t)r * p->mode, p->type); }   // rows: v rows of type bytes, mode apart
        else if (p->kind == 3) {
            if (p->addr) { const uint8_t *h0s = (const uint8_t *)(uintptr_t)p->addr, *g = p->dst; uint64_t lost = 0;   // H0: CPU writes since the dispatch was recorded, about to be overwritten
                           for (size_t q = 0; q + 4 <= p->n; q += 4) if (memcmp(g + q, h0s + q, 4) && memcmp(g + q, p->src + q, 4)) lost += 4;
                           atomic_fetch_add(&g_h0_lost_b, lost); atomic_fetch_add(&g_h0_lost_n, 1); free((void *)h0s); }
            if (chain_snap_mode()) { lk_(&g_gwmu, LW_GWMU); gw_snap_follow(p->dst, p->n, p->src, NULL, c->fence_val); pthread_mutex_unlock(&g_gwmu); }   // DM_VK_GW_CHAIN_SNAP
            memcpy(p->dst, p->src, p->n); hwl_add(p->dst, p->n);
            if (c->plog && p->n <= 64) { float f[16]; memcpy(f, p->src, p->n); LOG("plog: written back %#llx +%zu: %g %g %g %g%s", (unsigned long long)(uintptr_t)p->dst, p->n, f[0], p->n > 4 ? f[1] : 0, p->n > 8 ? f[2] : 0, p->n > 12 ? f[3] : 0, p->n > 16 ? " ..." : ""); } }
    }
    c->npost = 0;
    gw_expire(c->fence_val);
    lk_(&g_gwmu, LW_GWMU); uint64_t hold = c->hold; H0_GWH_T0();
    for (int i = 0; i < c->nextra; i++) {
        int used = hold > c->fence_val; for (int k = 0; k < g_ngw && !used; k++) used = g_gw[k].buf == c->extra[i].b;
        if (used) {   // a later submission reads it (gw_find), or a live entry points into it: freed once nothing does (xr_free)
            if (g_nxr == g_cxr) { g_cxr = g_cxr ? 2 * g_cxr : 16; g_xr = realloc(g_xr, g_cxr * sizeof *g_xr); }
            g_xr[g_nxr++] = (typeof(g_xr[0])){c->extra[i].b, c->extra[i].m, hold > c->fence_val ? hold : c->fence_val + 1, c->extra[i].size}; continue; }
        xring_put(c->extra[i].b, c->extra[i].m, c->extra[i].size); atomic_fetch_sub(&g_extra_ring_bytes, c->extra[i].size); }
    H0_GWH_END(1); pthread_mutex_unlock(&g_gwmu);
    c->nextra = 0;
    for (int i = 0; i < c->nfb; i++) vkDestroyFramebuffer(g_dev, c->fb[i], NULL);
    c->nfb = 0;
    for (int i = 0; i < c->nbv; i++) vkDestroyBufferView(g_dev, c->bv[i], NULL);
    c->nbv = 0;
    // The pools are idle now and the submit thread does not touch this context until g_done passes it (set after this returns): reset them here, off its time.
    // DM_VK_POOL_RESET=1 (vk19.64): the two command buffers one by one (the pool made with RESET_COMMAND_BUFFER_BIT), the whole pool when secondaries were used. Timed (the
    // profile put 1.56 ms a frame of dm-complete in vkResetCommandPool): the per-600 command-pool line
    if (!early_done_on()) { ctx_reset(c); c->reset_done = 1; }   // else after the fence is signalled (completion_thread)
}
static void ctx_reset(ctx_t *c) {
    { uint64_t t0 = now_ns();
      if (pool_reset_mode() == 1 && !c->nsec) { VKCHECK(vkResetCommandBuffer(c->cb, 0)); VKCHECK(vkResetCommandBuffer(c->up, 0)); atomic_fetch_add(&g_cpr_bufs, 1); }
      else VKCHECK(vkResetCommandPool(g_dev, c->pool, 0));
      atomic_fetch_add(&g_cpr_ns, now_ns() - t0); atomic_fetch_add(&g_cpr_n, 1); }
    for (int i = 0; i < c->ndp; i++) VKCHECK(vkResetDescriptorPool(g_dev, c->dps[i], 0));
}
// Lost device (a GPU hang the driver's watchdog ended): the work of every submission not known complete, so the log names the shaders and sizes that were running
static void trace_dump(const char *why) {
    LOG("[RED] vk: %s: the submissions in flight and what they held (newest last):", why);
    for (int k = 0; k < NCTX; k++) {
        ctx_t *c = &g_ctx[k]; if (c->fence_val <= g_done && c != C) continue;
        LOG("vk:  submission %llu%s: %u draws, %u dispatches, largest dispatch %u groups (%s %ux%ux%u)", (unsigned long long)c->fence_val, c == C ? " (being recorded)" : "", c->ndraw, c->ndisp,
            c->max_groups, c->heavy.pa ? c->heavy.pa : c->heavy.a, c->heavy.n0, c->heavy.n1, c->heavy.n2);
        uint32_t n = c->ntr < TRN ? c->ntr : TRN;
        for (uint32_t i = 0; i < n; i++) { const struct trace *t = &c->tr[(c->ntr - n + i) % TRN];
            const char *ta = t->pa ? t->pa : t->a, *tb = t->pa ? t->pb : t->b;
            if (t->kind == 'd') LOG("vk:   draw vs %s fs %s count %u x%u%s%s", ta, tb[0] ? tb : "-", t->n0, t->n1, t->gpu_ind ? " (indirect, GPU args)" : "", t->zero_tb ? " zero texel buffers" : "");
            else LOG("vk:   dispatch cs %s %ux%ux%u%s%s", ta, t->n0, t->n1, t->n2, t->gpu_ind ? " (indirect, GPU args)" : "", t->zero_tb ? " zero texel buffers" : ""); }
    }
}
static void trace_add(char kind, const char *a, const char *b, uint32_t n0, uint32_t n1, uint32_t n2, int gpu_ind, uint32_t zero_tb) {
    struct trace *t = &C->tr[C->ntr++ % TRN]; t->kind = kind; t->gpu_ind = (char)gpu_ind; t->zero_tb = zero_tb; t->n0 = n0; t->n1 = n1; t->n2 = n2;
    if (g_lean_tr) { t->pa = a ? a : "-"; t->pb = b ? b : ""; }   // DM_VK_LEAN: two pointers, the text only when a trace is dumped
    else { snprintf(t->a, sizeof t->a, "%s", a ? a : "-"); snprintf(t->b, sizeof t->b, "%s", b ? b : ""); t->pa = t->pb = NULL; }
    if (kind == 'd') { C->ndraw++; g_tex_draws_win++; } else { C->ndisp++; uint64_t g = (uint64_t)n0 * n1 * n2; if (g > C->max_groups) { C->max_groups = g > UINT32_MAX ? UINT32_MAX : (uint32_t)g; C->heavy = *t; } }
}
static void garbage_free(uint64_t done);
static void *completion_thread(void *a) { pthread_setname_np("dm-complete");
    (void)a;
    for (;;) {
        lk_(&g_mu, LW_MU);
        while (!g_npend) pthread_cond_wait(&g_cv, &g_mu);
        ctx_t *c = &g_ctx[g_pend[0]];
        pthread_mutex_unlock(&g_mu);
        VkResult r = vkWaitForFences(g_dev, 1, &c->fence, VK_TRUE, UINT64_MAX);
        if (r != VK_SUCCESS) { trace_dump("waiting for a submission failed"); die("Vulkan: waiting for a submission failed (%d)", (int)r); }
        uint64_t tf = now_ns();
        ctx_complete(c);
        int early = early_done_on();
        lk_(&g_mu, LW_MU);
        g_done = c->fence_val; memmove(g_pend, g_pend + 1, --g_npend * sizeof *g_pend);
        if (early) c->resetting = 1;
        pthread_cond_broadcast(&g_cv); pthread_mutex_unlock(&g_mu);
        atomic_fetch_add(&g_cl_ns, now_ns() - tf); atomic_fetch_add(&g_cl_n, 1);
        if (early) { ctx_reset(c); lk_(&g_mu, LW_MU); c->reset_done = 1; c->resetting = 0; pthread_cond_broadcast(&g_cv); pthread_mutex_unlock(&g_mu); }
        garbage_free(c->fence_val); xr_free(c->fence_val);
    }
    return NULL;
}

// ---------- Device ----------
static int g_has_u8;
static int g_pool_dev;   // the GPU can hold the pool set: partially bound, update-after-bind sampled images, non-uniform indexing (vk_pool)
static VkPhysicalDeviceFeatures g_feat;   // what the GPU has (every one of them is enabled at device creation): dualSrcBlend, wideLines, textureCompressionBC
static VkSampler g_dummy_samp; static VkImage g_dummy_img; static VkImageView g_dummy_view; static VkBuffer g_zero_buf, g_support_buf; static VkDeviceMemory g_zero_mem, g_support_mem;
// Stand-ins for a binding nothing real is bound to: a texel buffer of zeros (uniform), one for shader writes (storage, a scratch buffer of its own so the zeros stay zero), a storage image
static VkBufferView g_dummy_utb, g_dummy_stb; static VkBuffer g_scratch_buf; static VkDeviceMemory g_scratch_mem; static VkImage g_dummy_simg; static VkImageView g_dummy_simg_view;
static void dummies(void);
static int g_has_surface;
// Render passes: dynamic rendering where the GPU has it (Vulkan 1.3, or 1.2 / 1.1 with VK_KHR_dynamic_rendering), else classic render passes and framebuffers made per pass
// Every submission, the game's and the presents, waits on GPU for the one before and signals the next value of one timeline semaphore (g_tlv, under g_qmu). Submission order alone with a
// barrier at each start should do; the Mali-G57 (Unisoc T820) did not keep it for a present's blit, whose batch also waits on the swapchain acquire: two thirds of Metroid Dread's menu
// frames showed black. GPUs without timeline semaphores (Vulkan 1.1 without 1.2) wait for the GPU on the CPU before each present instead
static VkSemaphore g_tl; static uint64_t g_tlv; static int g_after_present = 1;   // g_after_present: the next game submission waits on the timeline (mtl_sub_commit)
// Pipelines compiled on earlier runs: the driver's pipeline cache, kept in <DM_VK_DIR>/../pipeline_cache.bin (written every 1800 frames when pipelines were added). A Mali-G57
// compiles one in about 40 ms, and a scene meeting a dozen new effects stalled its frame for 150-240 ms (Super Mario Odyssey); data from another GPU or driver is ignored by it
static VkPipelineCache g_pcache; static char g_pcache_path[1024]; static int g_pcache_saved;
static void pcache_load(void) {
    const char *d = getenv("DM_VK_DIR"); size_t n = 0; void *data = NULL;
    if (d) { snprintf(g_pcache_path, sizeof g_pcache_path, "%s/../pipeline_cache.bin", d); FILE *f = fopen(g_pcache_path, "rb");
             if (f) { fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET); if (sz > 0 && (data = malloc((size_t)sz)) && fread(data, 1, (size_t)sz, f) == (size_t)sz) n = (size_t)sz; fclose(f); } }
    VkPipelineCacheCreateInfo ci = {VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO, NULL, 0, n, data};
    if (vkCreatePipelineCache(g_dev, &ci, NULL, &g_pcache) != VK_SUCCESS) { ci.initialDataSize = 0; ci.pInitialData = NULL; if (vkCreatePipelineCache(g_dev, &ci, NULL, &g_pcache) != VK_SUCCESS) g_pcache = VK_NULL_HANDLE; }
    if (n) LOG("vk: pipeline cache of %zu KB loaded from an earlier run", n >> 10);
    free(data);
}
static int g_npso;
static void pcache_save(void) {
    if (!g_pcache || !g_pcache_path[0] || g_npso == g_pcache_saved) return;
    size_t n = 0; if (vkGetPipelineCacheData(g_dev, g_pcache, &n, NULL) != VK_SUCCESS || !n) return;
    void *data = malloc(n); if (!data) return;
    if (vkGetPipelineCacheData(g_dev, g_pcache, &n, data) == VK_SUCCESS) {
        char tmp[1100]; snprintf(tmp, sizeof tmp, "%s.tmp", g_pcache_path); FILE *f = fopen(tmp, "wb");
        if (f) { int ok = fwrite(data, 1, n, f) == n; ok &= fclose(f) == 0; if (ok) rename(tmp, g_pcache_path); }
        g_pcache_saved = g_npso;
    }
    free(data);
}
static int g_dynrend; static PFN_vkCmdBeginRenderingKHR p_begin_rendering; static PFN_vkCmdEndRenderingKHR p_end_rendering;
static VkDescriptorPool dpool_new(void);
static void ctx_init(int i) {   // a submission context's pool, command buffers, fence, ring, descriptor pool and query pools
        ctx_t *c = &g_ctx[i];
    // Reset as a whole each submission (ctx_complete / mtl_sub_begin), without releasing its memory. It was made TRANSIENT: a profile of FINAL FANTASY RESONANCE on the Adreno 825
    // showed a third of the render thread in kgsl_sharedmem_alloc ioctls under vkCmdCopyBufferToImage (the driver allocating command memory again as the buffers grew), which
    // a transient pool invites. DM_VK_POOL_TRANSIENT=1: as before
    static int tr = -1; if (tr < 0) { const char *e = getenv("DM_VK_POOL_TRANSIENT"); tr = e && atoi(e) != 0; LOG("vk: command pools %s (DM_VK_POOL_TRANSIENT=%d)", tr ? "transient" : "kept warm", tr); }
    VkCommandPoolCreateInfo pi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, (tr ? VK_COMMAND_POOL_CREATE_TRANSIENT_BIT : 0) | (pool_reset_mode() == 1 ? VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT : 0), g_qf};
    VKCHECK(vkCreateCommandPool(g_dev, &pi, NULL, &c->pool));
    VkCommandBuffer cbs[2]; VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, c->pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 2};
    VKCHECK(vkAllocateCommandBuffers(g_dev, &ai, cbs)); c->cb = cbs[0]; c->up = cbs[1];
    VkFenceCreateInfo fi = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, NULL, VK_FENCE_CREATE_SIGNALED_BIT};
    VKCHECK(vkCreateFence(g_dev, &fi, NULL, &c->fence));
    c->ring = c->ring0 = buf_new(RING_MIN, ALL_BUF_USAGE, &c->ringm, (void **)&c->ringp); c->ringm0 = c->ringm; c->ringp0 = c->ringp;
    c->ring0_size = c->ring_size = RING_MIN; atomic_fetch_add(&g_ring0_bytes, RING_MIN);
    c->dps = malloc(sizeof *c->dps); c->dps[0] = dpool_new(); c->ndp = 1;
    VkQueryPoolCreateInfo qi2 = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, NULL, 0, VK_QUERY_TYPE_TIMESTAMP, 3};   // the submission's start, its end, the upload command buffer's end
    VKCHECK(vkCreateQueryPool(g_dev, &qi2, NULL, &c->qp));
    VkQueryPoolCreateInfo qi3 = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, NULL, 0, VK_QUERY_TYPE_OCCLUSION, NOQ};
    VKCHECK(vkCreateQueryPool(g_dev, &qi3, NULL, &c->oq));
    VkQueryPoolCreateInfo qi4 = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, NULL, 0, VK_QUERY_TYPE_TIMESTAMP, NTQ};
    VKCHECK(vkCreateQueryPool(g_dev, &qi4, NULL, &c->tq));
    }
static void vk_init_(void) {
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "nxnative", 1, "dmhost", 1, VK_API_VERSION_1_3};
    const char *iext[] = {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_ANDROID_SURFACE_EXTENSION_NAME};
    uint32_t nie = 0; vkEnumerateInstanceExtensionProperties(NULL, &nie, NULL);
    VkExtensionProperties *iep = calloc(nie ? nie : 1, sizeof *iep); vkEnumerateInstanceExtensionProperties(NULL, &nie, iep);
    int nsurf = 0; for (uint32_t i = 0; i < nie; i++) for (int k = 0; k < 2; k++) nsurf += !strcmp(iep[i].extensionName, iext[k]);
    free(iep); g_has_surface = nsurf == 2;
    VkInstanceCreateInfo ii = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app, 0, NULL, g_has_surface ? 2 : 0, iext};
    VKCHECK(vkCreateInstance(&ii, NULL, &g_inst));
    uint32_t n = 1; VkResult r = vkEnumeratePhysicalDevices(g_inst, &n, &g_pd);
    if ((r != VK_SUCCESS && r != VK_INCOMPLETE) || !n) die("Vulkan: no GPU");
    vkGetPhysicalDeviceProperties(g_pd, &g_props); vkGetPhysicalDeviceMemoryProperties(g_pd, &g_memp);
    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(g_pd, &nq, NULL);
    VkQueueFamilyProperties qp[16]; if (nq > 16) nq = 16; vkGetPhysicalDeviceQueueFamilyProperties(g_pd, &nq, qp);
    for (g_qf = 0; g_qf < nq && !(qp[g_qf].queueFlags & VK_QUEUE_GRAPHICS_BIT); g_qf++) {}
    if (g_qf == nq) die("Vulkan: no graphics queue");
    // Every feature the GPU has is enabled (the translated shaders may use any of them), except the costly robustness beyond robustBufferAccess. The feature structures
    // chained are only those of the GPU's Vulkan version and extensions (1.1 phones have neither the 1.2 / 1.3 structures nor dynamic rendering)
    uint32_t dv = g_props.apiVersion;
    uint32_t ne = 0; vkEnumerateDeviceExtensionProperties(g_pd, NULL, &ne, NULL);
    VkExtensionProperties *ep = calloc(ne ? ne : 1, sizeof *ep); vkEnumerateDeviceExtensionProperties(g_pd, NULL, &ne, ep);
    int has_sc = 0, has_u8 = 0, has_dr = 0, has_ahb = 0, has_qff = 0;
    for (uint32_t i = 0; i < ne; i++) {
        has_sc |= !strcmp(ep[i].extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME); has_u8 |= !strcmp(ep[i].extensionName, VK_EXT_INDEX_TYPE_UINT8_EXTENSION_NAME);
        has_dr |= !strcmp(ep[i].extensionName, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME); g_has_budget |= !strcmp(ep[i].extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
        has_ahb |= !strcmp(ep[i].extensionName, VK_ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_EXTENSION_NAME); has_qff |= !strcmp(ep[i].extensionName, VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME);
    }
    {   // Reading an attachment inside its own pass (a held pass's depth sampled by a contact-shadow draw, now a split: vk19.4) needs one of these; logged to see which this GPU has
        static const char *const fl[] = {"VK_EXT_attachment_feedback_loop_layout", "VK_EXT_attachment_feedback_loop_dynamic_state", "VK_KHR_dynamic_rendering_local_read",
                                         "VK_EXT_rasterization_order_attachment_access", "VK_ARM_rasterization_order_attachment_access"};
        char b[512]; int o = 0;
        for (size_t k = 0; k < sizeof fl / sizeof *fl; k++) { uint32_t sv = 0; for (uint32_t i = 0; i < ne; i++) if (!strcmp(ep[i].extensionName, fl[k])) sv = ep[i].specVersion ? ep[i].specVersion : 1;
            o += snprintf(b + o, sizeof b - o, "%s%s %s", k ? ", " : "", fl[k], sv ? "yes" : "no"); }
        LOG("vk: in-pass attachment reads: %s (Vulkan %u.%u, driver %#x)", b, VK_API_VERSION_MAJOR(dv), VK_API_VERSION_MINOR(dv), g_props.driverVersion);
    }
    free(ep);
    VkPhysicalDeviceIndexTypeUint8FeaturesEXT u8 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INDEX_TYPE_UINT8_FEATURES_EXT};
    VkPhysicalDeviceDynamicRenderingFeaturesKHR dr = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR};
    VkPhysicalDeviceVulkan13Features f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan11Features f11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceShaderDrawParametersFeatures sdp = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES};
    void *chain = NULL;
    if (has_u8) { u8.pNext = chain; chain = &u8; }
    if (dv >= VK_API_VERSION_1_3) { f13.pNext = chain; chain = &f13; } else if (has_dr) { dr.pNext = chain; chain = &dr; }
    if (dv >= VK_API_VERSION_1_2) { f12.pNext = chain; chain = &f12; f11.pNext = chain; chain = &f11; } else { sdp.pNext = chain; chain = &sdp; }
    VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, chain};
    vkGetPhysicalDeviceFeatures2(g_pd, &f2);
    f13.robustImageAccess = VK_FALSE;
    g_feat = f2.features;
    g_pool_dev = dv >= VK_API_VERSION_1_2 && f12.descriptorBindingPartiallyBound && f12.descriptorBindingSampledImageUpdateAfterBind && f12.descriptorBindingUpdateUnusedWhilePending &&
                 f12.shaderSampledImageArrayNonUniformIndexing && g_feat.shaderSampledImageArrayDynamicIndexing;   // the pool set's needs (pool_dsl); every feature the GPU has is enabled anyway
    const char *ext[8]; uint32_t next = 0;
    if (g_has_budget) ext[next++] = VK_EXT_MEMORY_BUDGET_EXTENSION_NAME;
    if (has_sc && g_has_surface) ext[next++] = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
    if (has_u8 && u8.indexTypeUint8) { ext[next++] = VK_EXT_INDEX_TYPE_UINT8_EXTENSION_NAME; g_has_u8 = 1; }
    g_dynrend = dv >= VK_API_VERSION_1_3 ? f13.dynamicRendering : has_dr && dr.dynamicRendering;
    if (getenv("DM_VK_NO_DYNREND")) g_dynrend = 0;   // diagnostics: the render pass path of GPUs without dynamic rendering, on one that has it
    if (g_dynrend && dv < VK_API_VERSION_1_3) ext[next++] = VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME;
    if (zc_mode() && has_ahb && has_qff) { ext[next++] = VK_ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_EXTENSION_NAME; ext[next++] = VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME; g_zc_dev = 1; }
    float prio = 1;
    VkDeviceQueueCreateInfo qi = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, g_qf, 1, &prio};
    VkDeviceCreateInfo di = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2, 0, 1, &qi, 0, NULL, next, ext, NULL};
    VKCHECK(vkCreateDevice(g_pd, &di, NULL, &g_dev));
    vkGetDeviceQueue(g_dev, g_qf, 0, &g_q);
    pcache_load();
    if (dv >= VK_API_VERSION_1_2 && f12.timelineSemaphore && !getenv("DM_VK_NO_TIMELINE")) {
        VkSemaphoreTypeCreateInfo ti = {VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO, NULL, VK_SEMAPHORE_TYPE_TIMELINE, 0};
        VkSemaphoreCreateInfo si = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &ti, 0};
        if (vkCreateSemaphore(g_dev, &si, NULL, &g_tl) != VK_SUCCESS) g_tl = VK_NULL_HANDLE;
    }
    if (g_dynrend) {
        p_begin_rendering = (PFN_vkCmdBeginRenderingKHR)vkGetDeviceProcAddr(g_dev, dv >= VK_API_VERSION_1_3 ? "vkCmdBeginRendering" : "vkCmdBeginRenderingKHR");
        p_end_rendering = (PFN_vkCmdEndRenderingKHR)vkGetDeviceProcAddr(g_dev, dv >= VK_API_VERSION_1_3 ? "vkCmdEndRendering" : "vkCmdEndRenderingKHR");
        if (!p_begin_rendering || !p_end_rendering) g_dynrend = 0;
    }
    { const char *e = getenv("DM_VK_CTX"); if (e && *e) { int n = atoi(e); NCTX = n < 4 ? 4 : n > NCTX_MAX ? NCTX_MAX : n; } LOG("vk: %d submission contexts (DM_VK_CTX)", NCTX); }
    for (int i = 0; i < NCTX; i++) ctx_init(i);
    dummies();
    pthread_t th; if (pthread_create(&th, NULL, completion_thread, NULL)) die("Vulkan: cannot start the completion thread"); pthread_detach(th);
    LOG("vk: dual-source blending %s, wide lines %s (%.1f-%.1f px), BC textures %s", g_feat.dualSrcBlend ? "yes" : "no", g_feat.wideLines ? "yes" : "no", g_props.limits.lineWidthRange[0], g_props.limits.lineWidthRange[1],
        g_feat.textureCompressionBC ? "sampled by the GPU" : "decoded on upload");
    LOG("vk: %s, Vulkan %u.%u.%u, driver %#x; uint8 indices %s, %s, %s", g_props.deviceName, VK_API_VERSION_MAJOR(g_props.apiVersion), VK_API_VERSION_MINOR(g_props.apiVersion),
        VK_API_VERSION_PATCH(g_props.apiVersion), g_props.driverVersion, g_has_u8 ? "yes" : "no", g_dynrend ? "dynamic rendering" : "render passes (no dynamic rendering)",
        g_tl ? "submissions chained by a timeline semaphore" : "no timeline semaphore (presents wait for the GPU)");
}
static pthread_once_t g_once = PTHREAD_ONCE_INIT;
static void tog_init(void);
static void vk_init(void) { pthread_once(&g_once, vk_init_); tog_init(); }
static _Atomic int g_ndp_total;
// A descriptor pool of modest size: Unreal's shaders bind a dozen texel buffers, an array of 32 more and an array of 32 samplers a stage, so a submission with many draws chains
// several (dset_alloc); one pool sized for the worst case per context held hundreds of thousands of descriptors in eight contexts
static VkDescriptorPool dpool_new(void);
static VkDescriptorSet patch_set(void) {   // from the submission's descriptor pools, as write_set takes them
    VkDescriptorSet ds; VkDescriptorSetAllocateInfo ai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, C->dps[C->idp], 1, &g_patch_dsl};
    while (vkAllocateDescriptorSets(g_dev, &ai, &ds) != VK_SUCCESS) {
        if (++C->idp == C->ndp) { if (C->ndp >= 256) { C->idp--; return VK_NULL_HANDLE; } C->dps = realloc(C->dps, (C->ndp + 1) * sizeof *C->dps); C->dps[C->ndp++] = dpool_new(); }
        ai.descriptorPool = C->dps[C->idp];
    }
    return ds;
}
static VkDescriptorPool dpool_new(void) {
    VkDescriptorPoolSize ps[] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4096}, {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4096}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1024},
                                 {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 512}, {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 12288}, {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 512},
                                 {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1024}, {VK_DESCRIPTOR_TYPE_SAMPLER, 12288}, {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 4096}};
    VkDescriptorPoolCreateInfo dpi = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, 1024, sizeof ps / sizeof *ps, ps};
    VkDescriptorPool p; VKCHECK(vkCreateDescriptorPool(g_dev, &dpi, NULL, &p)); atomic_fetch_add(&g_ndp_total, 1); return p;
}

// One-time commands outside the submissions (dummy contents, the present readback): their own pool, the queue lock held, waited for
static VkCommandPool g_opool;
static VkCommandBuffer once_begin(void) {
    lk_(&g_qmu, LW_QMU);
    if (!g_opool) { VkCommandPoolCreateInfo pi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT, g_qf}; VKCHECK(vkCreateCommandPool(g_dev, &pi, NULL, &g_opool)); }
    VkCommandBuffer cb; VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, g_opool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    VKCHECK(vkAllocateCommandBuffers(g_dev, &ai, &cb));
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL};
    VKCHECK(vkBeginCommandBuffer(cb, &bi)); return cb;
}
static void once_end(VkCommandBuffer cb) {
    VKCHECK(vkEndCommandBuffer(cb));
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cb, 0, NULL};
    VKCHECK(vkQueueSubmit(g_q, 1, &si, VK_NULL_HANDLE)); VKCHECK(vkQueueWaitIdle(g_q));
    vkFreeCommandBuffers(g_dev, g_opool, 1, &cb);
    pthread_mutex_unlock(&g_qmu);
}
static uint64_t g_vbar_epoch;   // barriers that make attachment writes visible to vertex shaders (DM_VK_PASS_BARRIER): a texture marked in an older epoch is safe for them
static void barrier_all(VkCommandBuffer cb) {
    if (C && cb == C->cb) g_vbar_epoch++;
    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
}
// Scoped barriers in place of barrier_all (ALL_COMMANDS → ALL_COMMANDS, every access) after each dispatch and copy: FINAL FANTASY RESONANCE's battle frame has 77 dispatches, each
// followed by a full drain of the GPU; GameTurbo showed the Adreno 825 15% busy at 27 fps while the submissions spanned 24 ms
#define CONSUMER_STAGES (VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT | \
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT | \
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT)
#define CONSUMER_ACCESS (VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_INDEX_READ_BIT | VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | \
                         VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | \
                         VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT)
static VkPipelineStageFlags g_consumer_stages;   // CONSUMER_STAGES without the geometry stage on a GPU that has none
// A timed region: gpu_mark(kind, 0) before it, gpu_mark(kind, 1) after (kind 0 render pass, 1 dispatch, 2 copy / clear)
// Each region carries a signature (g_tsig_next: a pass's size, attachments and load actions, a dispatch's shader), and its GPU time is summed per signature (gsig_add, on the
// completion thread): the 600-frame lines name the passes and dispatches that cost the most (gsig_report)
static uint64_t g_tsig_next;
static void gpu_mark(int kind, int end) {
    if (C->ntq + 1 >= NTQ && !end) return;
    if (!end && (C->ntq & 1)) return;                       // a region already open
    if (end && !(C->ntq & 1)) return;                       // none open
    if (!end) { C->tkind[C->ntq / 2] = (uint8_t)kind; C->tsig[C->ntq / 2] = g_tsig_next; C->tdraws[C->ntq / 2] = 0; g_tsig_next = 0; }
    vkCmdWriteTimestamp(C->cb, end ? VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, C->tq, C->ntq++);
}
static void barrier_after(VkCommandBuffer cb, VkPipelineStageFlags src, VkAccessFlags src_access) {   // what src wrote, for everything recorded after it
    if (!g_consumer_stages) g_consumer_stages = CONSUMER_STAGES & ~(g_feat.geometryShader ? 0 : VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT);
    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, src_access, CONSUMER_ACCESS};
    vkCmdPipelineBarrier(cb, src, g_consumer_stages, 0, 1, &mb, 0, NULL, 0, NULL);
}
#define barrier_cs(cb) barrier_after(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT)
#define barrier_xfer(cb) barrier_after(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT)
static void to_general(VkCommandBuffer cb, VkImage img, VkImageAspectFlags asp) {
    VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, 0, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                              VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, img, {asp, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 1, &b);
}
static void dummies(void) {
    VkSamplerCreateInfo si = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO}; si.magFilter = si.minFilter = VK_FILTER_LINEAR; si.maxLod = 0;
    VKCHECK(vkCreateSampler(g_dev, &si, NULL, &g_dummy_samp));
    VkImageCreateInfo ii = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, {1, 1, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
                            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
    VKCHECK(vkCreateImage(g_dev, &ii, NULL, &g_dummy_img));
    VkMemoryRequirements r; vkGetImageMemoryRequirements(g_dev, g_dummy_img, &r); VkDeviceMemory m = mem_alloc(r, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VKCHECK(vkBindImageMemory(g_dev, g_dummy_img, m, 0));
    VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, g_dummy_img, VK_IMAGE_VIEW_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, {0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VKCHECK(vkCreateImageView(g_dev, &vi, NULL, &g_dummy_view));
    void *zp, *sp;
    g_zero_buf = buf_new(65536, ALL_BUF_USAGE, &g_zero_mem, &zp); memset(zp, 0, 65536);
    // Ryujinx's support buffer: alpha test 0, is_bgra[8] at 16, viewport inverse 144, viewport size 160, render scale count 176, render scale float4[73] at 192 (all 1.0)
    g_support_buf = buf_new(4096, ALL_BUF_USAGE, &g_support_mem, &sp); memset(sp, 0, 4096);
    for (int i = 0; i < 73; i++) *(float *)((uint8_t *)sp + 192 + 16 * i) = 1.0f;
    void *scp; g_scratch_buf = buf_new(65536, ALL_BUF_USAGE, &g_scratch_mem, &scp); memset(scp, 0, 65536);
    VkBufferViewCreateInfo bvi = {VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO, NULL, 0, g_zero_buf, VK_FORMAT_R32G32B32A32_SFLOAT, 0, 65536};
    VKCHECK(vkCreateBufferView(g_dev, &bvi, NULL, &g_dummy_utb));
    bvi.buffer = g_scratch_buf; VKCHECK(vkCreateBufferView(g_dev, &bvi, NULL, &g_dummy_stb));
    VkImageCreateInfo si2 = ii; si2.format = VK_FORMAT_R32G32B32A32_SFLOAT; si2.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    VKCHECK(vkCreateImage(g_dev, &si2, NULL, &g_dummy_simg));
    vkGetImageMemoryRequirements(g_dev, g_dummy_simg, &r); VkDeviceMemory m2 = mem_alloc(r, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT); VKCHECK(vkBindImageMemory(g_dev, g_dummy_simg, m2, 0));
    VkImageViewCreateInfo svi = vi; svi.image = g_dummy_simg; svi.format = VK_FORMAT_R32G32B32A32_SFLOAT; VKCHECK(vkCreateImageView(g_dev, &svi, NULL, &g_dummy_simg_view));
    VkCommandBuffer cb = once_begin(); to_general(cb, g_dummy_img, VK_IMAGE_ASPECT_COLOR_BIT); to_general(cb, g_dummy_simg, VK_IMAGE_ASPECT_COLOR_BIT);
    VkClearColorValue black = {{0, 0, 0, 1}}, zero = {{0}}; VkImageSubresourceRange rr = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(cb, g_dummy_img, VK_IMAGE_LAYOUT_GENERAL, &black, 1, &rr); vkCmdClearColorImage(cb, g_dummy_simg, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &rr); once_end(cb);
}

// ---------- Formats ----------
// Compressed formats Mali cannot sample, decoded on upload into the smallest format that samples the same: BC1-3 and BC7 to RGBA8, BC4 to R8, BC5 to RG8 (UNORM or SNORM: the
// missing channels read 0, alpha 1, as BC4 / BC5 sample), unsigned BC6H to B10G11R11 (its 16-bit floats lose only mantissa bits below what BC6H keeps), signed BC6H to RGBA16F.
// BC4-7 by bcdec (third_party/bcdec, MIT / Unlicense)
enum { DEC_NONE, DEC_BC1, DEC_BC2, DEC_BC3, DEC_BC4, DEC_BC4S, DEC_BC5, DEC_BC5S, DEC_BC7, DEC_BC6H, DEC_BC6HS };
static int dec_bpp(int dec) { return dec == DEC_BC4 || dec == DEC_BC4S ? 1 : dec == DEC_BC5 || dec == DEC_BC5S ? 2 : dec == DEC_BC6HS ? 8 : 4; }   // bytes per decoded pixel
typedef struct { VkFormat f; uint8_t noalpha, dec, depth, stencil; } fmt_t;
static fmt_t fmt_of(int nvn) {
    fmt_t r = {VK_FORMAT_UNDEFINED, 0, 0, 0, 0};
    if (nvn >= 0x79 && nvn <= 0x94) {   // ASTC: 4x4 5x4 5x5 6x5 6x6 8x5 8x6 8x8 10x5 10x6 10x8 10x10 12x10 12x12, LDR then sRGB; Vulkan lists UNORM / SRGB in pairs
        int k = (nvn - 0x79) % 14, srgb = nvn >= 0x87; r.f = (VkFormat)(VK_FORMAT_ASTC_4x4_UNORM_BLOCK + 2 * k + srgb); return r;
    }
    switch (nvn) {
        case 0x01: r.f = VK_FORMAT_R8_UNORM; break; case 0x02: r.f = VK_FORMAT_R8_SNORM; break; case 0x03: r.f = VK_FORMAT_R8_UINT; break; case 0x04: r.f = VK_FORMAT_R8_SINT; break;
        case 0x05: r.f = VK_FORMAT_R16_SFLOAT; break; case 0x06: r.f = VK_FORMAT_R16_UNORM; break; case 0x07: r.f = VK_FORMAT_R16_SNORM; break;
        case 0x08: r.f = VK_FORMAT_R16_UINT; break; case 0x09: r.f = VK_FORMAT_R16_SINT; break;
        case 0x0A: r.f = VK_FORMAT_R32_SFLOAT; break; case 0x0B: r.f = VK_FORMAT_R32_UINT; break; case 0x0C: r.f = VK_FORMAT_R32_SINT; break;
        case 0x0D: r.f = VK_FORMAT_R8G8_UNORM; break; case 0x0E: r.f = VK_FORMAT_R8G8_SNORM; break; case 0x0F: r.f = VK_FORMAT_R8G8_UINT; break; case 0x10: r.f = VK_FORMAT_R8G8_SINT; break;
        case 0x11: r.f = VK_FORMAT_R16G16_SFLOAT; break; case 0x12: r.f = VK_FORMAT_R16G16_UNORM; break; case 0x13: r.f = VK_FORMAT_R16G16_SNORM; break;
        case 0x14: r.f = VK_FORMAT_R16G16_UINT; break; case 0x15: r.f = VK_FORMAT_R16G16_SINT; break;
        case 0x16: r.f = VK_FORMAT_R32G32_SFLOAT; break; case 0x17: r.f = VK_FORMAT_R32G32_UINT; break; case 0x18: r.f = VK_FORMAT_R32G32_SINT; break;
        case 0x6D: r.noalpha = 1; /* fall through */ case 0x25: r.f = VK_FORMAT_R8G8B8A8_UNORM; break;
        case 0x6E: r.noalpha = 1; /* fall through */ case 0x26: r.f = VK_FORMAT_R8G8B8A8_SNORM; break;
        case 0x6F: r.noalpha = 1; /* fall through */ case 0x27: r.f = VK_FORMAT_R8G8B8A8_UINT; break;
        case 0x70: r.noalpha = 1; /* fall through */ case 0x28: r.f = VK_FORMAT_R8G8B8A8_SINT; break;
        case 0x71: r.noalpha = 1; /* fall through */ case 0x29: r.f = VK_FORMAT_R16G16B16A16_SFLOAT; break;
        case 0x72: r.noalpha = 1; /* fall through */ case 0x2A: r.f = VK_FORMAT_R16G16B16A16_UNORM; break;
        case 0x73: r.noalpha = 1; /* fall through */ case 0x2B: r.f = VK_FORMAT_R16G16B16A16_SNORM; break;
        case 0x74: r.noalpha = 1; /* fall through */ case 0x2C: r.f = VK_FORMAT_R16G16B16A16_UINT; break;
        case 0x75: r.noalpha = 1; /* fall through */ case 0x2D: r.f = VK_FORMAT_R16G16B16A16_SINT; break;
        case 0x76: r.noalpha = 1; /* fall through */ case 0x2E: r.f = VK_FORMAT_R32G32B32A32_SFLOAT; break;
        case 0x77: r.noalpha = 1; /* fall through */ case 0x2F: r.f = VK_FORMAT_R32G32B32A32_UINT; break;
        case 0x78: r.noalpha = 1; /* fall through */ case 0x30: r.f = VK_FORMAT_R32G32B32A32_SINT; break;
        case 0x31: r.f = VK_FORMAT_D24_UNORM_S8_UINT; r.depth = 1; r.stencil = 1; break;   // STENCIL8 has no sampled / attachment support on Mali: a D24S8 image's stencil
        case 0x32: r.f = VK_FORMAT_D16_UNORM; r.depth = 1; break;
        case 0x33: case 0x35: r.f = VK_FORMAT_D24_UNORM_S8_UINT; r.depth = 1; r.stencil = 1; break;
        case 0x34: r.f = VK_FORMAT_D32_SFLOAT; r.depth = 1; break;
        case 0x36: r.f = VK_FORMAT_D32_SFLOAT_S8_UINT; r.depth = 1; r.stencil = 1; break;
        case 0x37: r.noalpha = 1; /* fall through */ case 0x38: r.f = VK_FORMAT_R8G8B8A8_SRGB; break;
        case 0x39: r.f = VK_FORMAT_A4B4G4R4_UNORM_PACK16; break;
        case 0x3A: r.noalpha = 1; /* fall through */ case 0x3B: r.f = VK_FORMAT_A1R5G5B5_UNORM_PACK16; break;
        case 0x3C: r.f = VK_FORMAT_R5G6B5_UNORM_PACK16; break;
        case 0x3D: r.f = VK_FORMAT_A2B10G10R10_UNORM_PACK32; break; case 0x3E: r.f = VK_FORMAT_A2B10G10R10_UINT_PACK32; break;
        case 0x3F: r.f = VK_FORMAT_B10G11R11_UFLOAT_PACK32; break; case 0x40: r.f = VK_FORMAT_E5B9G9R9_UFLOAT_PACK32; break;
        case 0x41: r.noalpha = 1; /* fall through */ case 0x42: r.f = VK_FORMAT_R8G8B8A8_UNORM; r.dec = DEC_BC1; break;
        case 0x43: r.f = VK_FORMAT_R8G8B8A8_UNORM; r.dec = DEC_BC2; break;
        case 0x44: r.f = VK_FORMAT_R8G8B8A8_UNORM; r.dec = DEC_BC3; break;
        case 0x45: r.noalpha = 1; /* fall through */ case 0x46: r.f = VK_FORMAT_R8G8B8A8_SRGB; r.dec = DEC_BC1; break;
        case 0x47: r.f = VK_FORMAT_R8G8B8A8_SRGB; r.dec = DEC_BC2; break;
        case 0x48: r.f = VK_FORMAT_R8G8B8A8_SRGB; r.dec = DEC_BC3; break;
        case 0x49: r.f = VK_FORMAT_R8_UNORM; r.dec = DEC_BC4; break; case 0x4A: r.f = VK_FORMAT_R8_SNORM; r.dec = DEC_BC4S; break;
        case 0x4B: r.f = VK_FORMAT_R8G8_UNORM; r.dec = DEC_BC5; break; case 0x4C: r.f = VK_FORMAT_R8G8_SNORM; r.dec = DEC_BC5S; break;
        case 0x4D: r.f = VK_FORMAT_R8G8B8A8_UNORM; r.dec = DEC_BC7; break; case 0x4E: r.f = VK_FORMAT_R8G8B8A8_SRGB; r.dec = DEC_BC7; break;
        case 0x4F: r.f = VK_FORMAT_R16G16B16A16_SFLOAT; r.dec = DEC_BC6HS; break; case 0x50: r.f = VK_FORMAT_B10G11R11_UFLOAT_PACK32; r.dec = DEC_BC6H; break;
        case 0x99: r.noalpha = 1; /* fall through */ case 0x9A: r.f = VK_FORMAT_B8G8R8A8_UNORM; break;
        case 0x9B: r.noalpha = 1; /* fall through */ case 0x9C: r.f = VK_FORMAT_B8G8R8A8_SRGB; break;
        case 0x95: r.f = VK_FORMAT_B5G6R5_UNORM_PACK16; break;
        case 0x96: r.noalpha = 1; /* fall through */ case 0x97: r.f = VK_FORMAT_B5G5R5A1_UNORM_PACK16; break;
        case 0x98: r.f = VK_FORMAT_A1R5G5B5_UNORM_PACK16; break;
    }
    // A GPU that samples BC (textureCompressionBC: Adreno, most desktop-class GPUs) takes the blocks as they are: no decode on upload, and a quarter to an eighth of the memory
    if (r.dec && g_feat.textureCompressionBC) {
        static const VkFormat NAT[16] = {VK_FORMAT_BC1_RGB_UNORM_BLOCK, VK_FORMAT_BC1_RGBA_UNORM_BLOCK, VK_FORMAT_BC2_UNORM_BLOCK, VK_FORMAT_BC3_UNORM_BLOCK, VK_FORMAT_BC1_RGB_SRGB_BLOCK,
            VK_FORMAT_BC1_RGBA_SRGB_BLOCK, VK_FORMAT_BC2_SRGB_BLOCK, VK_FORMAT_BC3_SRGB_BLOCK, VK_FORMAT_BC4_UNORM_BLOCK, VK_FORMAT_BC4_SNORM_BLOCK, VK_FORMAT_BC5_UNORM_BLOCK,
            VK_FORMAT_BC5_SNORM_BLOCK, VK_FORMAT_BC7_UNORM_BLOCK, VK_FORMAT_BC7_SRGB_BLOCK, VK_FORMAT_BC6H_SFLOAT_BLOCK, VK_FORMAT_BC6H_UFLOAT_BLOCK};   // NVN 0x41 … 0x50
        if (nvn >= 0x41 && nvn <= 0x50) { r.f = NAT[nvn - 0x41]; r.dec = DEC_NONE; }
    }
    return r;
}
static int fmt_block(VkFormat f) { return f >= VK_FORMAT_BC1_RGB_UNORM_BLOCK && f <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK; }   // BC, ETC2, EAC, ASTC
// Vertex attribute formats (NVN format → Vulkan, *_UI2F / *_I2F → SCALED), 0 = none
static VkFormat vfmt(int f) {
    static const VkFormat T[0x6D] = {
        [0x01] = VK_FORMAT_R8_UNORM, VK_FORMAT_R8_SNORM, VK_FORMAT_R8_UINT, VK_FORMAT_R8_SINT, VK_FORMAT_R16_SFLOAT, VK_FORMAT_R16_UNORM, VK_FORMAT_R16_SNORM, VK_FORMAT_R16_UINT, VK_FORMAT_R16_SINT,
        VK_FORMAT_R32_SFLOAT, VK_FORMAT_R32_UINT, VK_FORMAT_R32_SINT, VK_FORMAT_R8G8_UNORM, VK_FORMAT_R8G8_SNORM, VK_FORMAT_R8G8_UINT, VK_FORMAT_R8G8_SINT,
        VK_FORMAT_R16G16_SFLOAT, VK_FORMAT_R16G16_UNORM, VK_FORMAT_R16G16_SNORM, VK_FORMAT_R16G16_UINT, VK_FORMAT_R16G16_SINT, VK_FORMAT_R32G32_SFLOAT, VK_FORMAT_R32G32_UINT, VK_FORMAT_R32G32_SINT,
        VK_FORMAT_R8G8B8_UNORM, VK_FORMAT_R8G8B8_SNORM, VK_FORMAT_R8G8B8_UINT, VK_FORMAT_R8G8B8_SINT, VK_FORMAT_R16G16B16_SFLOAT, VK_FORMAT_R16G16B16_UNORM, VK_FORMAT_R16G16B16_SNORM,
        VK_FORMAT_R16G16B16_UINT, VK_FORMAT_R16G16B16_SINT, VK_FORMAT_R32G32B32_SFLOAT, VK_FORMAT_R32G32B32_UINT, VK_FORMAT_R32G32B32_SINT,
        VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SNORM, VK_FORMAT_R8G8B8A8_UINT, VK_FORMAT_R8G8B8A8_SINT, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_UNORM,
        VK_FORMAT_R16G16B16A16_SNORM, VK_FORMAT_R16G16B16A16_UINT, VK_FORMAT_R16G16B16A16_SINT, VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R32G32B32A32_UINT, VK_FORMAT_R32G32B32A32_SINT,
        [0x3D] = VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_FORMAT_A2B10G10R10_UINT_PACK32, VK_FORMAT_B10G11R11_UFLOAT_PACK32,
        [0x51] = VK_FORMAT_R8_USCALED, VK_FORMAT_R8_SSCALED, VK_FORMAT_R16_USCALED, VK_FORMAT_R16_SSCALED, VK_FORMAT_R32_UINT, VK_FORMAT_R32_SINT,
        VK_FORMAT_R8G8_USCALED, VK_FORMAT_R8G8_SSCALED, VK_FORMAT_R16G16_USCALED, VK_FORMAT_R16G16_SSCALED, VK_FORMAT_R32G32_UINT, VK_FORMAT_R32G32_SINT,
        VK_FORMAT_R8G8B8_USCALED, VK_FORMAT_R8G8B8_SSCALED, VK_FORMAT_R16G16B16_USCALED, VK_FORMAT_R16G16B16_SSCALED, VK_FORMAT_R32G32B32_UINT, VK_FORMAT_R32G32B32_SINT,
        VK_FORMAT_R8G8B8A8_USCALED, VK_FORMAT_R8G8B8A8_SSCALED, VK_FORMAT_R16G16B16A16_USCALED, VK_FORMAT_R16G16B16A16_SSCALED, VK_FORMAT_R32G32B32A32_UINT, VK_FORMAT_R32G32B32A32_SINT,
        [0x69] = VK_FORMAT_A2B10G10R10_SNORM_PACK32, VK_FORMAT_A2B10G10R10_SINT_PACK32, VK_FORMAT_A2B10G10R10_USCALED_PACK32, VK_FORMAT_A2B10G10R10_SSCALED_PACK32,
    };
    return f > 0 && f < 0x6D ? T[f] : VK_FORMAT_UNDEFINED;
}

// BC1 / BC2 / BC3 → RGBA8 (Mali has no BC sampling); BC4-7 by bcdec
#define BCDEC_IMPLEMENTATION
#define BCDEC_BC4BC5_PRECISE
#define BCDECDEF static
#include "../third_party/bcdec/bcdec.h"
static void bc_color(const uint8_t *b, uint8_t out[16][4], int alpha1) {
    uint16_t c0 = (uint16_t)(b[0] | b[1] << 8), c1 = (uint16_t)(b[2] | b[3] << 8); uint8_t p[4][4];
    for (int i = 0; i < 2; i++) { uint16_t c = i ? c1 : c0; p[i][0] = (uint8_t)((c >> 11) * 255 / 31); p[i][1] = (uint8_t)((c >> 5 & 63) * 255 / 63); p[i][2] = (uint8_t)((c & 31) * 255 / 31); p[i][3] = 255; }
    if (c0 > c1 || !alpha1) for (int k = 0; k < 3; k++) { p[2][k] = (uint8_t)((2 * p[0][k] + p[1][k]) / 3); p[3][k] = (uint8_t)((p[0][k] + 2 * p[1][k]) / 3); }
    else for (int k = 0; k < 3; k++) { p[2][k] = (uint8_t)((p[0][k] + p[1][k]) / 2); p[3][k] = 0; }
    p[2][3] = 255; p[3][3] = c0 > c1 || !alpha1 ? 255 : 0;
    uint32_t idx = (uint32_t)b[4] | (uint32_t)b[5] << 8 | (uint32_t)b[6] << 16 | (uint32_t)b[7] << 24;
    for (int i = 0; i < 16; i++) memcpy(out[i], p[idx >> (2 * i) & 3], 4);
}
static void bc_spread(uint8_t px[16][8]) { for (int i = 15; i >= 0; i--) memmove(px[i], (uint8_t *)px + 4 * i, 4); }   // 16 packed RGBA8 pixels → one per 8-byte slot
static void bc_decode(int dec, const uint8_t *src, size_t src_pitch, uint32_t w, uint32_t h, uint8_t *dst) {   // dst: w * h * dec_bpp(dec)
    uint32_t bw = (w + 3) / 4, bh = (h + 3) / 4, bs = dec == DEC_BC1 || dec == DEC_BC4 || dec == DEC_BC4S ? 8 : 16, pb = (uint32_t)dec_bpp(dec);
    for (uint32_t by = 0; by < bh; by++) for (uint32_t bx = 0; bx < bw; bx++) {
        const uint8_t *b = src + by * src_pitch + bx * bs; uint8_t px[16][8];
        if (dec == DEC_BC1) bc_color(b, (uint8_t (*)[4])px, 1), bc_spread(px);
        else if (dec == DEC_BC4 || dec == DEC_BC4S || dec == DEC_BC5 || dec == DEC_BC5S) {
            int sg = dec == DEC_BC4S || dec == DEC_BC5S, two = dec == DEC_BC5 || dec == DEC_BC5S; uint8_t rg[16][2] = {{0}}, r1[16] = {0};
            if (two) bcdec_bc5(b, rg, 8, sg); else bcdec_bc4(b, r1, 4, sg);   // pitch in bytes: rows of 4 RG / R pixels
            for (int i = 0; i < 16; i++) { px[i][0] = two ? rg[i][0] : r1[i]; px[i][1] = two ? rg[i][1] : 0; px[i][2] = 0; px[i][3] = sg ? 127 : 255; }
        } else if (dec == DEC_BC7) bcdec_bc7(b, px, 32);   // RGBA8 rows of 4 pixels, in a block buffer of 8-byte pixels: pitch 32 bytes
        else if (dec == DEC_BC6H || dec == DEC_BC6HS) {
            uint16_t rgb[16][3]; bcdec_bc6h_half(b, rgb, 12, dec == DEC_BC6HS);   // pitch in halves
            for (int i = 0; i < 16; i++) {
                if (dec == DEC_BC6HS) { uint16_t q[4] = {rgb[i][0], rgb[i][1], rgb[i][2], 0x3C00}; memcpy(px[i], q, 8); continue; }
                // half → unsigned 11 / 10-bit floats: same 5-bit exponent and bias, the mantissa cut to 6 / 5 bits; negative values (none in unsigned BC6H) as 0
                uint32_t r11 = rgb[i][0] & 0x8000 ? 0 : rgb[i][0] >> 4, g11 = rgb[i][1] & 0x8000 ? 0 : rgb[i][1] >> 4, b10 = rgb[i][2] & 0x8000 ? 0 : rgb[i][2] >> 5;
                uint32_t pk = (r11 & 0x7ff) | (g11 & 0x7ff) << 11 | (b10 & 0x3ff) << 22; memcpy(px[i], &pk, 4);
            }
        } else {
            bc_color(b + 8, (uint8_t (*)[4])px, 0); bc_spread(px);
            if (dec == DEC_BC2) for (int i = 0; i < 16; i++) { int a = b[i / 2] >> (4 * (i & 1)) & 15; px[i][3] = (uint8_t)(a * 17); }
            else {
                uint8_t a[8] = {b[0], b[1]};
                if (a[0] > a[1]) for (int i = 2; i < 8; i++) a[i] = (uint8_t)(((8 - i) * a[0] + (i - 1) * a[1]) / 7);
                else { for (int i = 2; i < 6; i++) a[i] = (uint8_t)(((6 - i) * a[0] + (i - 1) * a[1]) / 5); a[6] = 0; a[7] = 255; }
                uint64_t bits = 0; for (int i = 0; i < 6; i++) bits |= (uint64_t)b[2 + i] << (8 * i);
                for (int i = 0; i < 16; i++) px[i][3] = a[bits >> (3 * i) & 7];
            }
        }
        for (int y = 0; y < 4; y++) for (int x = 0; x < 4; x++) {
            uint32_t X = bx * 4 + x, Y = by * 4 + y;
            if (X < w && Y < h) memcpy(dst + ((size_t)Y * w + X) * pb, dec == DEC_BC7 ? (const uint8_t *)px + y * 32 + x * 4 : px[y * 4 + x], pb);
        }
    }
}

// ---------- Images ----------
typedef struct { uint64_t key; VkImageView v; } vview;
struct vtex {
    uint64_t serial; VkImage img; VkDeviceMemory m; int blk; VkDeviceSize moff, msize; fmt_t f; int nvn_fmt, target;
    uint32_t w, h, d, levels, layers; VkImageAspectFlags aspect; int uploaded, gpu;
    int dirty;   // the CPU wrote its storage since the last upload (WriteTexels, a movie frame, a flushed pool range): uploaded again at its next use, unless the GPU renders into it
    const uint8_t *mem; uint64_t len;   // its storage in guest memory
    uint64_t reup_frame;   // the frame of its last upload (re-uploads at most once a frame, tex_get)
    uint64_t clean_gen;    // the submission (g_bvgen) in which its storage was last found unwritten since its upload (DM_VK_DIRTY_ONCE)
    uint64_t stamp; const void *span;   // uploaded from a dirty-tracked pool (nvn_dirty.c): the write generation then and the tracked range, re-uploaded when the game writes it
    vview *views; int nviews;
    struct vtex *next;
    uint64_t dead;   // released by the game: freed once the GPU has finished the submission that was being recorded then (fence dead)
    uint64_t lhash, lhash_gen; uint8_t lcpu;
    uint8_t ubwc;   // made compressible (DM_VK_UBWC)
    VkFormat vfl[2]; uint8_t nvfl; VkImageFormatListCreateInfo fli;   // DM_VK_UBWC: the only formats its views may take (nvfl 0: any, MUTABLE_FORMAT without a list)
    VkImageCreateInfo ci; uint8_t packaged, evicted; uint64_t last_frame, last_fence;   // texture eviction (tex_evict): how it was made, whether its content comes from its storage, when it was last used
    uint32_t gpu_lv;   // levels a copy wrote on the GPU (texture or buffer copies into one level): an upload from storage leaves them alone, the rest still follows the game's writes   // lcpu: a linear texture in a pool the CPU writes (a glyph cache): its storage hashed once a submission, uploaded again when it changed
    uint64_t dfr_gen;   // a storage image a deferred dispatch writes (defer generation g_defer_gen while it is pending)
    uint64_t vhaz_epoch; uint8_t vhaz;   // DM_VK_PASS_BARRIER: written by a pass whose end barrier left vertex shaders out, in that barrier epoch (see pass_barrier_on)
    int storage;   // made with VK_IMAGE_USAGE_STORAGE_BIT (an NVN image texture whose format Vulkan can store to)
    int copied;   // a GPU copy (copy_b2t) or a shader's image store wrote into it: hibernation reads it back before a snapshot even when it was uploaded first
    uint8_t late;   // made before the game wrote its storage, watched for that (late_fill)
    uint32_t mir_lv;   // levels a GPU copy wrote whose bytes were written into the storage too (copy_b2t / copy_t2t mirror): the storage holds them, as the image does
    int inpool; uint64_t pool_out;   // entries of the pool set holding a view of it (vk_pool: never evicted, not freed while held); the submission after which the last one let go
    uint32_t linrow, linbpb;   // a linear 2D texture (LINEAR / LINEAR_RENDER_TARGET) the CPU can read: its row pitch and bytes per texel; what passes render into it goes back to its storage (lin_writeback)
};
static vtex **g_gb; static int g_ngb, g_cgb; static pthread_mutex_t g_gbmu = PTHREAD_MUTEX_INITIALIZER;
#define NHASH 4096
static vtex *g_tex[NHASH];
static vtex *tex_find(uint64_t ser) { for (vtex *v = g_tex[ser % NHASH]; v; v = v->next) if (v->serial == ser) return v; return NULL; }
static uint64_t *g_cpu_written; static int g_ncpu, g_ccpu;   // serials WriteTexels wrote (mtl_tex_cpu_written)
static int cpu_written(uint64_t ser) { for (int i = 0; i < g_ncpu; i++) if (g_cpu_written[i] == ser) return 1; return 0; }

static int g_plog;   // DM_VK_PASSLOG=<frame>[+]: that frame's (or every frame's from there) passes, draws, copies, clears and uploads are logged (diagnostics)
// Diagnostics: DM_VK_SKIP=<mask> leaves out kinds of GPU work, to find which one a lost device comes from: 1 buffer-to-texture copies, 2 texture copies, 4 texture-to-buffer
// copies, 8 texture clears, 16 texture uploads, 32 draws, 64 attachment clears
enum { SK_B2T = 1, SK_T2T = 2, SK_T2B = 4, SK_CLEAR = 8, SK_UPLOAD = 16, SK_DRAW = 32, SK_ACLEAR = 64 };
static int vk_skip(void) { static int m = -1; if (m < 0) { const char *e = getenv("DM_VK_SKIP"); m = e ? (int)strtol(e, NULL, 0) : 0; if (m) LOG("DM_VK_SKIP=%#x: GPU work of those kinds is left out", m); } return m; }
static void tex_upload(vtex *v, const tex_t *t);
static void tex_reupload(vtex *v, const tex_t *t, uint64_t since);
// A texture with no packaged layout in a CPU_NO_ACCESS pool whose storage already holds data takes it as its initial content, as nvn_mtl.m tex_init_fill: games load a file into
// memory and make a GPU-only pool over it (Metroid Dread's BC6H light probe cubemaps, which sampled as zero and left every lit surface black). The same in a CPU_UNCACHED pool
// (its writes reach memory unflushed) and for storage the CPU flushed before the texture's first use (nvn_flushed; DM_NVN_FLUSH_FILL=0 leaves it out), with COMPRESSIBLE
// textures too: compression applies to what the GPU renders into one, and at its first use the storage holds plain texels (FINAL FANTASY XII THE ZODIAC AGE sets the flag on
// every texture, its CPU-written and flushed battle target ring and line art included, which drew with alpha 0 otherwise). Multisample and depth textures stay GPU textures
// DM_VK_UBWC=1: images whose views never reinterpret their format keep the Adreno's bandwidth compression (UBWC). Every image used to be made MUTABLE_FORMAT with no format list,
// which the driver may answer by leaving the image uncompressed: every render target, depth buffer and texture read and written at full size. Now an image is MUTABLE_FORMAT
// only with a format list -- its own format and its sRGB / UNORM sibling -- or not mutable at all where it has no sibling (depth images included: their views keep their format).
// Storage images stay as they were (any view). A view asking for a format outside the list takes the image's own (counted: g_ubwc_fallback). Default 0 (off), and to stay so:
// it bought no frame rate, and with it on FINAL FANTASY RESONANCE's texture streaming (100-300 MB a second of copies into compressed images) cost the system memory no resident set
// or device allocation showed -- MemAvailable 1.8 GB to 0.24 GB in 3 s, the low-memory killer 4 times in 4 at the Earth Shrine arch; with it off the same walk peaked 1.1 GB/s of uploads and lived
// DM_VK_UBWC=2 (vk19.35, opt-in): compression only for images made as render targets (first use an attachment) or marked COMPRESSIBLE on a format the GPU renders, and only when
// nothing is uploaded into them: not packaged, not CPU-written, no initial or late fill. Uploaded textures stay plain, the copies into which cost the untracked memory above.
// Copies into a compressed image are still counted (g_ubwc_b2t: the game's buffer-to-texture copies, g_ubwc_t2t: image copies and blits, g_ubwc_up: uploads, none expected)
static int ubwc_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_UBWC"); v = e ? atoi(e) : 0; if (v < 0 || v > 2) v = 0;
    if (v) LOG("DM_VK_UBWC=%d: %s keep compression (format lists)", v, v == 2 ? "render targets nothing is uploaded into" : "images without format reinterpretation"); } return v; }
static uint64_t g_ubwc_fallback, g_ubwc_imgs, g_ubwc_b2t, g_ubwc_t2t, g_ubwc_up;
static VkFormat fmt_sibling(VkFormat f) {   // the sRGB / UNORM twin of a format, VK_FORMAT_UNDEFINED when it has none
    static const VkFormat pr[][2] = {{VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SRGB}, {VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_B8G8R8A8_SRGB}, {VK_FORMAT_A8B8G8R8_UNORM_PACK32, VK_FORMAT_A8B8G8R8_SRGB_PACK32},
        {VK_FORMAT_R8_UNORM, VK_FORMAT_R8_SRGB}, {VK_FORMAT_R8G8_UNORM, VK_FORMAT_R8G8_SRGB}, {VK_FORMAT_R8G8B8_UNORM, VK_FORMAT_R8G8B8_SRGB},
        {VK_FORMAT_BC1_RGB_UNORM_BLOCK, VK_FORMAT_BC1_RGB_SRGB_BLOCK}, {VK_FORMAT_BC1_RGBA_UNORM_BLOCK, VK_FORMAT_BC1_RGBA_SRGB_BLOCK}, {VK_FORMAT_BC2_UNORM_BLOCK, VK_FORMAT_BC2_SRGB_BLOCK},
        {VK_FORMAT_BC3_UNORM_BLOCK, VK_FORMAT_BC3_SRGB_BLOCK}, {VK_FORMAT_BC7_UNORM_BLOCK, VK_FORMAT_BC7_SRGB_BLOCK},
        {VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK}, {VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK, VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK}, {VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK, VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK},
        {VK_FORMAT_ASTC_4x4_UNORM_BLOCK, VK_FORMAT_ASTC_4x4_SRGB_BLOCK}, {VK_FORMAT_ASTC_5x5_UNORM_BLOCK, VK_FORMAT_ASTC_5x5_SRGB_BLOCK}, {VK_FORMAT_ASTC_6x6_UNORM_BLOCK, VK_FORMAT_ASTC_6x6_SRGB_BLOCK},
        {VK_FORMAT_ASTC_8x8_UNORM_BLOCK, VK_FORMAT_ASTC_8x8_SRGB_BLOCK}};
    for (size_t i = 0; i < sizeof pr / sizeof *pr; i++) { if (pr[i][0] == f) return pr[i][1]; if (pr[i][1] == f) return pr[i][0]; }
    return VK_FORMAT_UNDEFINED;
}
static int mem_any(const uint8_t *p, uint64_t n) { uint64_t i = 0, x; for (; i + 8 <= n; i += 8) { memcpy(&x, p + i, 8); if (x) return 1; } for (; i < n; i++) if (p[i]) return 1; return 0; }
int nvn_flushed(const void *p, uint64_t n);   // hle_nvn.c
static int init_fill(const tex_t *t, const fmt_t *f) {
    if (!t->mem || !t->pool || (t->pool->flags & NVN_MEMORY_POOL_FLAGS_VIRTUAL) || t->has_layout) return 0;
    if (t->d.samples > 1 || f->depth || t->d.target == NVN_TEXTURE_TARGET_BUFFER) return 0;
    int no_cpu = (t->pool->flags & NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS) != 0, uncached = (t->pool->flags & NVN_MEMORY_POOL_FLAGS_CPU_UNCACHED) != 0;
    static int flush_fill = -1; if (flush_fill < 0) { const char *e = getenv("DM_NVN_FLUSH_FILL"); flush_fill = !e || atoi(e) != 0; }
    if (!no_cpu && !uncached && (!flush_fill || !nvn_flushed(t->mem, t->L.size))) return 0;
    return mem_any(t->mem, t->L.size);
}
// A texture init_fill would take from its storage, but whose storage is still all zero at its first use: the game's loader has not written it yet (FINAL FANTASY RESONANCE's
// castle materials on the Xiaomi Pad 8, whose file reads are slower than the Mac's: drawn before their data arrived). It used to stay a cleared GPU texture that nothing
// watched, and when the data came the material stayed black (a black albedo under the usual specular: dark, glossy columns and panels). Now its storage is watched from that
// first use and it is uploaded once the game writes it (tex_get), as a texture made from storage is. Not for render targets, images or CPU_NO_ACCESS pools (the CPU writes none).
// DM_VK_LATE_FILL=0 leaves it out
static uint64_t g_late_watch, g_late_fill;
static int late_fill(const tex_t *t, const fmt_t *f) {
    static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_LATE_FILL"); on = !e || atoi(e) != 0; }
    if (!on || !t->mem || !t->pool || (t->pool->flags & (NVN_MEMORY_POOL_FLAGS_VIRTUAL | NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS)) || t->has_layout) return 0;
    if (t->d.samples > 1 || f->depth || t->d.target == NVN_TEXTURE_TARGET_BUFFER) return 0;
    if (t->d.flags & (NVN_TEXTURE_FLAGS_IMAGE | NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET)) return 0;
    if ((t->d.flags & NVN_TEXTURE_FLAGS_COMPRESSIBLE) && !fmt_block(f->f) && !f->dec) return 0;   // COMPRESSIBLE marks a render target, except on a block-compressed format
    return !mem_any(t->mem, t->L.size);
}
// The content hash of a CPU-written linear texture (tex_get: once a submission for each one bound). Four independent lanes of the old step, folded at the end: the one chain
// waited on each multiply before the next (about 2 bytes a cycle), and Tomodachi Life's island spent 1.8 G cycles of a 20 s profile in it, 18% of the recording thread
// (vk-t8). Any change still changes the result: each step is a bijection of its lane, and so is the fold of each lane. g_lcpu_*: the per-600 line
static uint64_t g_lcpu_hashes, g_lcpu_bytes, g_lcpu_changed, g_lcpu_tracked;
static uint64_t g_lcpu_gated;   // lookups of a CPU-written linear texture left to the dirty pages instead of a hash (DM_VK_LCPU_DIRTY)
static int lcpu_dirty_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_LCPU_DIRTY"); v = !(e && atoi(e) == 0); LOG("vk: DM_VK_LCPU_DIRTY=%d: CPU-written linear textures in tracked storage %s", v, v ? "follow their dirty pages (no hash)" : "hashed once a submission"); } return v; }   // g_lcpu_tracked: hashes of storage the dirty pages cover (nvn_dirty_tracked; whether a page check could stand in, vk-t10)
static uint64_t mem_hash(const uint8_t *p, uint64_t n) {
    const uint64_t P = 0x100000001b3ull; uint64_t h0 = 0x9e3779b97f4a7c15ull, h1 = h0 ^ 1, h2 = h0 ^ 2, h3 = h0 ^ 3, x0, x1, x2, x3, i = 0;
    g_lcpu_hashes++; g_lcpu_bytes += n;
    for (; i + 32 <= n; i += 32) { memcpy(&x0, p + i, 8); memcpy(&x1, p + i + 8, 8); memcpy(&x2, p + i + 16, 8); memcpy(&x3, p + i + 24, 8);
                                   h0 = (h0 ^ x0) * P; h1 = (h1 ^ x1) * P; h2 = (h2 ^ x2) * P; h3 = (h3 ^ x3) * P; }
    uint64_t h = (((h0 * P) ^ h1) * P ^ h2) * P ^ h3;
    for (; i + 8 <= n; i += 8) { memcpy(&x0, p + i, 8); h = (h ^ x0) * P; }
    for (; i < n; i++) h = (h ^ p[i]) * P;
    return h;
}
static int g_occ_over;
static _Atomic uint64_t g_presents;   // frames presented (mtl_present)
static uint64_t g_texget_n;   // tex_get calls (the vk19.59 counters: per draw; a plain counter, the recording thread)
static uint64_t g_rcore[8];   // the recording thread's core at each submission's start (sched_getcpu; the vk19.59 counters)
static uint64_t g_evicted, g_remade;   // textures evicted / made again from their storage (the memory line)
static int tex_remake(vtex *v, const tex_t *t);
static uint64_t g_pkg_up, g_pkg_reup;   // textures uploaded from their storage when made / again after the game wrote it (the packaged textures line, mtl_pkg_stats)
static uint64_t g_bvgen;   // the submission's number (bview, the linear texture hashes)
static uint64_t g_tex_dead, g_treg_dead;   // lookups of a texture object that is no longer one (tex_get) / registrations of one (mtl_reg_tex)
// Re-uploads (vk19.34). FINAL FANTASY RESONANCE streams an area's textures into their pool storage while drawing them: each write brought a re-upload of the whole texture at its
// next use, up to once per submission (8.7 a frame), 77-289 MB a second through the rings, and the low-memory killer while the system lost memory nobody's resident set showed.
// Now: only the levels written since the last upload go again (DM_VK_REUP_PARTIAL, default 1); a texture is re-uploaded at most once a frame, and while memory is short
// (g_mem_low) at most every DM_VK_REUP_FRAMES_LOW frames (default 4) and only until a submission has DM_VK_REUP_CAP_MB (default 16) of uploads; held ones keep their old
// content until then. First uploads are never held
static uint64_t g_reup_held, g_reup_lv_skipped; static _Atomic uint64_t g_up_ring_cum;
static int reup_partial(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_REUP_PARTIAL"); v = !e || atoi(e) != 0; } return v; }
static int reup_hold(const vtex *v, uint64_t fr) {
    static long lowf = -1, cap = -1;
    if (lowf < 0) { const char *e = getenv("DM_VK_REUP_FRAMES_LOW"); lowf = e ? atol(e) : 4; e = getenv("DM_VK_REUP_CAP_MB"); cap = (e ? atol(e) : 16) << 20; }
    if (!v->reup_frame) return 0;
    if (fr == v->reup_frame) return 1;   // already uploaded this frame
    if (g_mem_low && (fr - v->reup_frame < (uint64_t)lowf || (cap > 0 && g_sub_up_bytes >= (uint64_t)cap))) return 1;
    return 0;
}
// DM_VK_DIRTY_ONCE (vk19.37, default 1): whether a packaged texture's storage was written since its upload is asked once a submission, not at every binding. The question scans
// the generation of every page the texture covers (a 5 MB texture: 1,300 pages), and the dungeon's ~1,000 draws a frame bind several textures each: thousands of scans a frame.
// Found clean, a texture stays clean for the rest of the submission's recording: the GPU runs the submission later anyway, and a write landing meanwhile is seen by the next one
static uint64_t g_dirty_scans, g_dirty_skips;
static int dirty_once(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_DIRTY_ONCE"); v = !e || atoi(e) != 0; } return v; }
static int reup_this_frame(const vtex *v) {   // uploaded already this frame: a re-upload would be held (reup_hold), so the scan is left out
    if (!dirty_once() || v->late || v->lcpu || !v->reup_frame || v->reup_frame != atomic_load(&g_presents)) return 0;   // (a CPU-written linear texture is never held: DM_VK_LCPU_DIRTY)
    g_dirty_skips++; return 1;
}
static int tex_clean_now(vtex *v, const tex_t *t) {
    g_dirty_scans++;
    if (nvn_dirty_since_span(v->span, t->mem, t->L.size, v->stamp)) return 0;
    if (dirty_once()) v->clean_gen = g_bvgen;
    return 1;
}
static vtex *tex_get(const tex_t *t, int for_rt) {
    g_texget_n++;   // calls per draw (the vk19.59 counters)
    // A texture object that is no longer one -- finalized, its memory reused for other data -- is never made into an image (as nvn_mtl.m tex_mtl_ checks the magic): FINAL
    // FANTASY RESONANCE registers texture objects whose memory holds half-float constants by then (serial 0x3c003c003c003c00), and their slots and image bindings stay bound.
    // An unknown format used to stop them by chance; a plausible one would have made an image of garbage size from a garbage storage pointer
    if (!t || t->magic != MAGIC('T', 'E', 'X', 0)) { if (t) g_tex_dead++; return NULL; }
    lk_(&g_rmu, LW_RMU); vtex *v = tex_find(t->serial); pthread_mutex_unlock(&g_rmu);
    // Not while evicted (tex_evict, the low-memory trim): its image is gone, and the remake below uploads it from its storage anyway. Tomodachi Life crashed in the driver's
    // vkCmdCopyBufferToImage2 two frames after a trim evicted 58 textures: this upload went into VK_NULL_HANDLE (FINAL FANTASY RESONANCE's CPU-written atlases are used every frame)
    // DM_VK_LCPU_DIRTY (vk-t12, default 1): a CPU-written linear texture in dirty-tracked storage (v->span, set when tex_upload_ stamped and protected it) is not hashed:
    // a write to its pages makes it dirty and the re-upload below takes it, as for packaged textures but never held (reup_hold, reup_this_frame and the HTAB / stage-reuse
    // fast paths exempt it), so a write is uploaded at the next submission that uses it, as the hash did (Unreal's Slate atlas gains glyphs mid-frame; ffr-perf's review). Tomodachi Life hashed one 16 MB texture a frame that never changed
    // (0.58 G cycles in 20 s, ~2 ms of recording); untracked storage is still hashed. DM_VK_LCPU_DIRTY=0: hashed always, as before
    if (v && v->lcpu && v->span && lcpu_dirty_on()) g_lcpu_gated++;
    else if (v && v->lcpu && !v->gpu && !v->evicted && !for_rt && v->lhash_gen != g_bvgen) {   // a CPU-written linear texture (Unreal's Slate font atlas, a glyph cache): the GPU reads it in place on the Switch
        v->lhash_gen = g_bvgen; if (nvn_dirty_tracked(t->mem, t->L.size)) g_lcpu_tracked++; uint64_t h = mem_hash(t->mem, t->L.size);
        if (h != v->lhash) { v->lhash = h; tex_upload(v, t); g_lcpu_changed++; }
    }
    if (v) {
        if (v->dead && v->dead < g_fence + 1) v->dead = g_fence + 1;   // released, still named by a command recorded before: kept until this submission is done
        v->last_frame = atomic_load(&g_presents); v->last_fence = g_fence + 1;
        if (v->evicted && !tex_remake(v, t)) return NULL;   // evicted while unused (tex_evict): made again from its storage
        if (for_rt) v->gpu = 1;
        else if (!v->gpu && t->mem && (v->dirty || (v->span && v->clean_gen != g_bvgen && !reup_this_frame(v) && !tex_clean_now(v, t)))) {
            uint64_t fr = atomic_load(&g_presents);
            if (!v->dirty && !v->late && !v->lcpu && reup_hold(v, fr)) g_reup_held++;   // its stamp still says written: the next use tries again (never a CPU-written linear texture: DM_VK_LCPU_DIRTY keeps the hash's per-submission freshness)
            else { uint64_t since = v->dirty ? 0 : v->stamp; v->dirty = 0; v->reup_frame = fr; tex_reupload(v, t, since); g_pkg_reup++;
                   if (v->late) { v->late = 0; v->packaged = 1; g_late_fill++; } } }   // a texture whose data came after its first use (late_fill): from its storage from now on   // never over what the GPU drew: a flush of the pool range around a render target is not new content
        return v;
    }
    fmt_t f = fmt_of(t->d.format);
    if (!f.f) { static int n; if (n++ < 8) LOG("[RED] vk: texture format %#x has no Vulkan format yet (serial %llu)", t->d.format, (unsigned long long)t->serial); return NULL; }
    if (t->d.target == NVN_TEXTURE_TARGET_BUFFER) { RED_ONCE(3, "buffer textures are not implemented yet"); return NULL; }
    atomic_fetch_add(&g_tex_made, 1);
    v = calloc(1, sizeof *v); v->serial = t->serial; v->f = f; v->nvn_fmt = t->d.format; v->target = t->d.target; v->mem = t->mem; v->len = t->L.size;
    v->w = t->d.width ? t->d.width : 1; v->h = t->d.height ? t->d.height : 1; v->levels = t->d.levels ? t->d.levels : 1;
    int is3d = t->d.target == NVN_TEXTURE_TARGET_3D, cube = t->d.target == NVN_TEXTURE_TARGET_CUBEMAP || t->d.target == NVN_TEXTURE_TARGET_CUBEMAP_ARRAY;
    v->d = is3d ? (t->d.depth ? t->d.depth : 1) : 1;
    v->layers = is3d ? 1 : t->L.layers ? t->L.layers : t->d.target == NVN_TEXTURE_TARGET_2D_ARRAY || t->d.target == NVN_TEXTURE_TARGET_1D_ARRAY ? (t->d.depth ? t->d.depth : 1) : cube ? 6 : 1;
    if (t->d.target == NVN_TEXTURE_TARGET_1D_ARRAY && !t->L.layers) v->layers = t->d.height ? t->d.height : 1;
    v->aspect = f.depth ? VK_IMAGE_ASPECT_DEPTH_BIT | (f.stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : 0) : VK_IMAGE_ASPECT_COLOR_BIT;
    VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties(g_pd, f.f, &fp);
    VkImageUsageFlags u = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) u |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) u |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    if ((fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) && (t->d.flags & NVN_TEXTURE_FLAGS_IMAGE)) u |= VK_IMAGE_USAGE_STORAGE_BIT;
    VkSampleCountFlagBits sc = t->d.samples > 1 ? (VkSampleCountFlagBits)t->d.samples : VK_SAMPLE_COUNT_1_BIT;
    v->storage = (u & VK_IMAGE_USAGE_STORAGE_BIT) != 0;
    VkImageCreateInfo ii = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | (cube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0) | (is3d ? VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT : 0),
                            is3d ? VK_IMAGE_TYPE_3D : t->d.target == NVN_TEXTURE_TARGET_1D || t->d.target == NVN_TEXTURE_TARGET_1D_ARRAY ? VK_IMAGE_TYPE_1D : VK_IMAGE_TYPE_2D,
                            f.f, {v->w, t->d.target == NVN_TEXTURE_TARGET_1D_ARRAY ? 1 : v->h, v->d}, v->levels, v->layers, sc, VK_IMAGE_TILING_OPTIMAL, u, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
    if (sc != VK_SAMPLE_COUNT_1_BIT) ii.mipLevels = 1;
    int want_ubwc = ubwc_on() && !(u & VK_IMAGE_USAGE_STORAGE_BIT);
    if (want_ubwc && ubwc_on() == 2) {   // render targets only, and nothing uploaded into them
        int rt = for_rt || ((t->d.flags & NVN_TEXTURE_FLAGS_COMPRESSIBLE) && !fmt_block(f.f) && !f.dec && (u & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)));
        int uploads = !for_rt && t->mem && (t->has_layout || cpu_written(t->serial) || (t->d.flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET)) || init_fill(t, &f) || late_fill(t, &f));
        want_ubwc = rt && !uploads;
    }
    if (want_ubwc) {   // see ubwc_on: a format list, or not mutable
        v->ubwc = 1;
        VkFormat sib = f.depth ? VK_FORMAT_UNDEFINED : fmt_sibling(f.f); v->vfl[0] = f.f; v->nvfl = 1;
        if (sib != VK_FORMAT_UNDEFINED) { v->vfl[1] = sib; v->nvfl = 2; v->fli = (VkImageFormatListCreateInfo){VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO, NULL, 2, v->vfl}; ii.pNext = &v->fli; }
        else ii.flags &= ~(VkImageCreateFlags)VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
        g_ubwc_imgs++;
    }
    VkResult r = vkCreateImage(g_dev, &ii, NULL, &v->img);
    if (r != VK_SUCCESS) { LOG("[RED] vk: image %ux%ux%u x%u levels %u format %#x target %d not created (%d)", v->w, v->h, v->d, v->layers, v->levels, t->d.format, t->d.target, (int)r); free(v); return NULL; }
    VkMemoryRequirements mr; vkGetImageMemoryRequirements(g_dev, v->img, &mr); VkDeviceSize off;
    v->blk = img_mem(mr, &v->m, &off); v->moff = off; v->msize = mr.size; VKCHECK(vkBindImageMemory(g_dev, v->img, v->m, off));
    if ((t->d.flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET)) && t->mem && t->d.target == NVN_TEXTURE_TARGET_2D && v->levels == 1 && v->layers == 1 && !f.dec && !f.depth && sc == VK_SAMPLE_COUNT_1_BIT) {
        int bw = 1, bh = 1, bpb = nvn_fmt_bpb(t->d.format, &bw, &bh); uint32_t row = t->d.stride ? (uint32_t)t->d.stride : v->w * (uint32_t)bpb;
        if (bpb && bw == 1 && bh == 1 && row % bpb == 0 && (uint64_t)row * v->h <= t->L.size) { v->linrow = row; v->linbpb = (uint32_t)bpb; }
    }
    to_general(C->up, v->img, v->aspect); C->up_used = 1;
    v->gpu = for_rt;
    lk_(&g_rmu, LW_RMU); v->next = g_tex[v->serial % NHASH]; g_tex[v->serial % NHASH] = v; pthread_mutex_unlock(&g_rmu);
    // A linear 2D texture in a pool the CPU can write (not CPU_NO_ACCESS, at most 16 MB): its storage is what the game wrote there, uploaded now and again whenever it changes
    v->lcpu = (t->d.flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET)) && t->mem && t->pool && !(t->pool->flags & NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS) &&
              !(t->pool->flags & NVN_MEMORY_POOL_FLAGS_VIRTUAL) && t->L.size <= (16u << 20) && !f.depth && sc == VK_SAMPLE_COUNT_1_BIT;
    int packaged = (t->has_layout || cpu_written(t->serial) || init_fill(t, &f) || v->lcpu) && t->mem && !for_rt;
    { static int tl = -1; if (tl < 0) { const char *e = getenv("DM_VK_TEXLOG"); tl = e && atoi(e) ? atoi(e) : 0; } static int n;   // diagnostics: DM_VK_TEXLOG=<n> logs the first n textures made
      if (tl && n++ < tl) LOG("vk: texture %llu: %ux%ux%u x%u levels %u format %#x target %d flags %#x stride %u size %llu pool %#x%s%s%s%s", (unsigned long long)t->serial, v->w, v->h, v->d, v->layers, v->levels,
                              t->d.format, t->d.target, t->d.flags, t->d.stride, (unsigned long long)t->L.size, t->pool ? t->pool->flags : 0, t->has_layout ? " packaged" : "", v->lcpu ? " linear-cpu" : "",
                              packaged ? " uploaded" : "", for_rt ? " render-target" : ""); }
    if (v->lcpu && packaged) { v->lhash = mem_hash(t->mem, t->L.size); v->lhash_gen = g_bvgen; }
    v->ci = ii; v->packaged = packaged && !for_rt; v->last_frame = atomic_load(&g_presents); v->last_fence = g_fence + 1;
    if (packaged) { tex_upload(v, t); g_pkg_up++; v->reup_frame = atomic_load(&g_presents); }
    else if (!for_rt && late_fill(t, &f) && nvn_dirty_tracked(t->mem, t->L.size)) {   // still empty: cleared now, watched, uploaded when the game writes it (late_fill)
        v->stamp = nvn_dirty_now(); nvn_dirty_protect(t->mem, t->L.size); v->span = nvn_dirty_span(t->mem, t->L.size); v->late = 1; g_late_watch++;
        VkClearColorValue z = {{0}}; VkImageSubresourceRange rr = {VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
        if (!fmt_block(f.f) && (u & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) { vkCmdClearColorImage(C->up, v->img, VK_IMAGE_LAYOUT_GENERAL, &z, 1, &rr); C->up_used = 1; }
    }
    else {   // contents come from the GPU (copies, render passes): start cleared, as a new texture's storage would read on hardware after a clear. Render targets too: a pass that
             // loads one before anything drew into it, or a window texture presented with nothing drawn (FINAL FANTASY XII THE ZODIAC AGE's first frames; magenta on the Metal host
             // until it zero-filled them), would read Vulkan's undefined content
        VkClearColorValue z = {{0}}; VkImageSubresourceRange rr = {VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
        if (!f.depth && !fmt_block(f.f) && sc == VK_SAMPLE_COUNT_1_BIT && (u & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) { vkCmdClearColorImage(C->up, v->img, VK_IMAGE_LAYOUT_GENERAL, &z, 1, &rr); C->up_used = 1; }   // no compressed format (vkCmdClearColorImage)
    }
    return v;
}
// ---------- Texture eviction ----------
// The images of packaged textures mirror what the game holds in its own memory pools (the storage they were uploaded from), so one unused for a long while is freed and made again
// from that storage at its next use: half a gigabyte of images in FINAL FANTASY RESONANCE's battles, most of them for places the camera had left. Only textures whose content is
// their storage (packaged, never rendered into or copied to on the GPU, not written by the CPU since), and only when no submission in flight can use them. While the image memory
// is above EVICT_FLOOR, those unused for EVICT_AGE frames go; on Android's low-memory signal (dm_vk_trim), those unused for a second
#define EVICT_FLOOR (128ull << 20)
#define EVICT_AGE 900
static int tex_remake(vtex *v, const tex_t *t) {
    if (vkCreateImage(g_dev, &v->ci, NULL, &v->img) != VK_SUCCESS) { v->img = VK_NULL_HANDLE; RED_ONCE(90, "an evicted texture could not be made again"); return 0; }
    VkMemoryRequirements mr; vkGetImageMemoryRequirements(g_dev, v->img, &mr); VkDeviceSize off;
    v->blk = img_mem(mr, &v->m, &off); v->moff = off; v->msize = mr.size; VKCHECK(vkBindImageMemory(g_dev, v->img, v->m, off));
    to_general(C->up, v->img, v->aspect); C->up_used = 1;
    v->evicted = 0; g_remade++;
    v->gpu_lv &= ~v->mir_lv;   // levels a GPU copy wrote come back from the storage they were mirrored into
    tex_upload(v, t);
    return 1;
}
static uint64_t g_trim_gen_tex;
static _Atomic uint64_t g_reg_gen;   // any texture or sampler registration, or a texture release (the stage-level set reuse, DM_VK_STAGE_REUSE)
// What a pool set (vk_pool; one per pool base, POOL_SETS) must look at again: the IDs registered since its last sync (a bitmap per set, under g_rmu), textures released or
// written by the CPU while held, sampler registrations; g_pool_live = the sets made so far
#define POOL_SETS 4
static uint64_t *g_pool_dbits[POOL_SETS]; static int g_pool_ndw[POOL_SETS]; static int g_pool_live;
static _Atomic uint64_t g_pool_kill, g_pool_cpu, g_pool_sgen; static _Atomic int g_tpool_n;
static void pool_mark_set(int k, int id) {   // caller holds g_rmu
    int w = id >> 6; if (w >= g_pool_ndw[k]) { int n = w + 64; g_pool_dbits[k] = realloc(g_pool_dbits[k], n * sizeof **g_pool_dbits); memset(g_pool_dbits[k] + g_pool_ndw[k], 0, (n - g_pool_ndw[k]) * sizeof **g_pool_dbits); g_pool_ndw[k] = n; }
    g_pool_dbits[k][w] |= 1ull << (id & 63);
}
static void pool_mark(int id) { for (int k = 0; k < g_pool_live; k++) pool_mark_set(k, id); }   // caller holds g_rmu
// The age a trim evicts at: 60 frames, and DM_VK_EVICT_FAST (default 30) on an early trim for fast heap growth (mem_watch). FINAL FANTASY RESONANCE's battles load about 300 MB
// of the game heap at the transition, while the field's textures sit unused behind the battle
static _Atomic int g_trim_age = 60;
static void tex_evict(uint64_t trim_gen) {
    static uint64_t last; uint64_t n = atomic_load(&g_presents); int trim = trim_gen != g_trim_gen_tex; g_trim_gen_tex = trim_gen;
    if (!trim && n - last < 60) return;
    last = n;
    uint64_t held = 0; lk_(&g_blkmu, LW_BLKMU); for (int i = 0; i < g_nblk; i++) if (g_blk[i].m && g_blk[i].live) held += g_blk[i].size; pthread_mutex_unlock(&g_blkmu);   // the empty spare (DM_VK_SPARE_BLOCK) not counted
    if (trim) blk_spare_trim();
    if (!trim && held < EVICT_FLOOR) return;
    static int64_t eage = -1; if (eage < 0) { const char *e = getenv("DM_VK_EVICT_AGE"); eage = e && atoll(e) > 0 ? atoll(e) : 300; }
    uint64_t age = trim ? (uint64_t)atomic_load(&g_trim_age) : (uint64_t)eage, done; lk_(&g_mu, LW_MU); done = g_done; pthread_mutex_unlock(&g_mu);
    int nev = 0;
    lk_(&g_rmu, LW_RMU);
    for (int hh = 0; hh < NHASH; hh++) for (vtex *v = g_tex[hh]; v; v = v->next) {
        if (v->dead || v->evicted || v->inpool || !v->packaged || v->gpu || (v->gpu_lv & ~v->mir_lv) || v->dirty || !v->img || v->last_fence > done || n - v->last_frame < age) continue;   // levels GPU copies wrote count when their bytes were mirrored into the storage (mir_lv)
        for (int k = 0; k < v->nviews; k++) vkDestroyImageView(g_dev, v->views[k].v, NULL);
        free(v->views); v->views = NULL; v->nviews = 0;
        vkDestroyImage(g_dev, v->img, NULL); img_mem_free(v->blk, v->m, v->moff, v->msize); v->img = VK_NULL_HANDLE; v->evicted = 1; nev++;
    }
    pthread_mutex_unlock(&g_rmu);
    g_evicted += nev;
    if (nev) { static int nl; if (nl++ < 8 || trim) LOG("vk: %d textures unused for %llu frames evicted%s (image memory was %.0f MB)", nev, (unsigned long long)age, trim ? " on the low-memory signal" : "", held / 1048576.0); }
}
// Packaged texture: each level and layer deswizzled from the pool storage into the ring, then copied into the image by the upload command buffer (it runs before this submission's draws)
static uint64_t g_t_up, g_up_bytes; uint64_t g_nsub_stat;
static void tex_upload_(vtex *v, const tex_t *t, uint64_t since);
static void tex_upload(vtex *v, const tex_t *t) { uint64_t t0 = now_ns(); tex_upload_(v, t, 0); g_t_up += now_ns() - t0; g_up_bytes += t->L.size; atomic_fetch_add(&g_up_cum, t->L.size); }
static void tex_reupload(vtex *v, const tex_t *t, uint64_t since) { uint64_t t0 = now_ns(); tex_upload_(v, t, reup_partial() ? since : 0); g_t_up += now_ns() - t0; g_up_bytes += t->L.size; atomic_fetch_add(&g_up_cum, t->L.size); }
// The compressed texels of a level the CPU decodes (BC / ASTC the GPU cannot sample) were deswizzled into a fresh malloc each time: at that size fresh mapped pages, faulted in on
// first write. DM_VK_DECODE_POOL=1 keeps one buffer (the largest needed so far, the submit thread's) instead. Default 0: as before
static uint8_t *g_dec_buf; static size_t g_dec_cap;
static int decode_pool(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_DECODE_POOL"); v = e && atoi(e) != 0; if (v) LOG("DM_VK_DECODE_POOL=1: one kept buffer for texture decoding"); } return v; }
static uint8_t *dec_tmp(size_t n) {
    if (!decode_pool()) return malloc(n);
    if (n > g_dec_cap) { uint8_t *b = realloc(g_dec_buf, n); if (!b) return malloc(n); g_dec_buf = b; g_dec_cap = n; }
    return g_dec_buf;
}
static void dec_tmp_done(uint8_t *p) { if (p != g_dec_buf) free(p); }
static void tex_upload_(vtex *v, const tex_t *t, uint64_t since) {
    if (vk_skip() & SK_UPLOAD) return;
    if (g_plog) LOG("plog:  upload %llu fmt %#x %ux%ux%u levels %u layers %u target %d", (unsigned long long)t->serial, t->d.format, v->w, v->h, v->d, v->levels, v->layers, t->d.target);
    int bw = 1, bh = 1, bpb = nvn_fmt_bpb(t->d.format, &bw, &bh);
    if (!bpb) { RED_ONCE(4, "texture format %#x: block size unknown, not uploaded", t->d.format); return; }
    // As nvn_mtl.m does for packaged textures: the storage is write-protected, and a game write to it (streamed texture data arriving after the texture was made, as Super Mario
    // Odyssey's normal maps do; noise until then) brings a new upload at the texture's next use (tex_get). The stamp is taken and the pages protected BEFORE the storage is read
    // (nvn_dirty.h): protected after the read, a write landing while it ran, or on a page a write had already unprotected, was never seen, and the texture kept what it was mid-write
    if (t->mem && !v->gpu && nvn_dirty_tracked(t->mem, t->L.size)) { v->stamp = nvn_dirty_now(); nvn_dirty_protect(t->mem, t->L.size); v->span = nvn_dirty_span(t->mem, t->L.size); }
    else since = 0;
    // A re-upload (since: the stamp of the last upload) takes only the levels whose storage was written after it, checked after the protect above so that a write racing it
    // is either seen here or faults again later. A page shared by two levels marks both. Arrays of more than 16 layers go whole
    uint32_t lvmask = ~0u;
    if (since && v->levels > 1 && v->levels <= 32 && v->layers <= 16) {
        lvmask = 0; uint64_t lsize = t->L.layer_size ? t->L.layer_size : t->L.size;
        for (unsigned lv = 0; lv < v->levels; lv++) {
            uint64_t a = nvn_tex_level_offset(&t->d, lv), b = lv + 1 < v->levels ? nvn_tex_level_offset(&t->d, lv + 1) : lsize;
            if (b <= a || b > lsize) { lvmask |= 1u << lv; continue; }
            for (unsigned ly = 0; ly < v->layers; ly++) if ((uint64_t)ly * lsize + b <= t->L.size && nvn_dirty_since(t->mem + (uint64_t)ly * lsize + a, b - a, since)) { lvmask |= 1u << lv; break; }
            if (!(lvmask >> lv & 1)) g_reup_lv_skipped++;
        }
    }
    // Every level and layer goes in one vkCmdCopyBufferToImage (regions batched while they share a ring): one call per level and layer cost a driver call each, and the Adreno's
    // allocated command memory inside them (kgsl_sharedmem_alloc, a third of the render thread while a stage streamed its textures in)
    enum { NUREG = 64 }; VkBufferImageCopy ureg[NUREG]; uint32_t nureg = 0; VkBuffer ubuf = VK_NULL_HANDLE;
    #define UFLUSH() do { if (nureg) { if (v->ubwc) g_ubwc_up++; vkCmdCopyBufferToImage(C->up, ubuf, v->img, VK_IMAGE_LAYOUT_GENERAL, nureg, ureg); nureg = 0; } } while (0)
    for (unsigned lv = 0; lv < v->levels; lv++) {
        if (lv < 32 && (v->gpu_lv >> lv & 1)) continue;   // a copy on the GPU wrote this level: its storage does not hold that
        if (lv < 32 && !(lvmask >> lv & 1)) continue;     // a re-upload: not written since the last one
        uint32_t w = v->w >> lv ? v->w >> lv : 1, h = v->h >> lv ? v->h >> lv : 1, d = v->d >> lv ? v->d >> lv : 1;
        if (t->d.target == NVN_TEXTURE_TARGET_1D_ARRAY) h = 1;
        uint32_t bx = (w + bw - 1) / bw, by = (h + bh - 1) / bh; size_t pitch = (size_t)bx * bpb, lsz = pitch * by * d;
        for (unsigned ly = 0; ly < v->layers; ly++) {
            size_t out = v->f.dec ? (size_t)w * h * dec_bpp(v->f.dec) * d : lsz;
            VkDeviceSize o = ring_alloc(out, 16); g_sub_up_bytes += out; atomic_fetch_add(&g_up_ring_cum, out);   // may move to an extra ring: the regions so far stay with theirs (ubuf, flushed below)
            if (o == VK_WHOLE_SIZE) { UFLUSH(); v->dirty = 1; return; }   // not uploaded: again at the next use (the stamp above would say nothing changed)
            uint8_t *dst = C->ringp + o, *tmp = v->f.dec ? dec_tmp(lsz) : dst;
            int rc = 0;
            if (t->d.flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET)) {
                // Linear storage: rows at the texture's own pitch (its stride, padded past width * bytes per texel), layers L.layer_size apart -- copied row by row. One block
                // of lsz bytes read the rows skewed whenever the pitch was padded, and LINEAR_RENDER_TARGET went through the block-linear deswizzle (glyphs drawn as blocky noise)
                size_t srow = t->d.levels <= 1 && t->d.height && t->L.layer_size >= (uint64_t)pitch * by ? (size_t)(t->L.layer_size / (by * d)) : pitch;
                if (lv == 0 && t->d.stride >= pitch) srow = t->d.stride;
                if (srow < pitch) srow = pitch;
                const uint8_t *src = t->mem + (uint64_t)ly * t->L.layer_size + nvn_tex_level_offset(&t->d, lv);
                if ((uint64_t)ly * t->L.layer_size + nvn_tex_level_offset(&t->d, lv) + srow * (by * d - 1) + pitch > t->L.size) { rc = 1; RED_ONCE(6, "linear texture %ux%u format %#x: rows at pitch %zu run past its storage", v->w, v->h, t->d.format, srow); }
                else for (uint32_t yy = 0; yy < by * d; yy++) memcpy(tmp + (size_t)yy * pitch, src + (size_t)yy * srow, pitch);
            } else rc = nvn_deswizzle(&t->d, lv, ly, t->mem, tmp, pitch);
            if (rc) { RED_ONCE(5, "texture format %#x target %d: deswizzle failed", t->d.format, t->d.target); if (v->f.dec) dec_tmp_done(tmp); continue; }
            if (v->f.dec) { for (uint32_t z = 0; z < d; z++) bc_decode(v->f.dec, tmp + pitch * by * z, pitch, w, h, dst + (size_t)w * h * dec_bpp(v->f.dec) * z); dec_tmp_done(tmp); }
            VkBufferImageCopy c = {o, 0, 0, {v->aspect & VK_IMAGE_ASPECT_COLOR_BIT ? VK_IMAGE_ASPECT_COLOR_BIT : VK_IMAGE_ASPECT_DEPTH_BIT, lv, ly, 1}, {0, 0, 0}, {w, h, d}};
            if (ubuf != C->ring) { UFLUSH(); ubuf = C->ring; }
            ureg[nureg++] = c; if (nureg == NUREG) UFLUSH();
        }
    }
    UFLUSH();
    #undef UFLUSH
    C->up_used = 1; v->uploaded = 1;
}
static void tex_free(vtex *v) {   // released by the game: draws already recorded (and the one being recorded) may still use it
    atomic_fetch_add_explicit(&g_reg_gen, 1, memory_order_relaxed);
    pthread_mutex_lock(&g_gbmu);
    v->dead = g_fence + 1;
    if (__atomic_load_n(&v->inpool, __ATOMIC_ACQUIRE)) atomic_fetch_add(&g_pool_kill, 1);   // the pool set lets go of it at its next sync
    if (g_ngb == g_cgb) { g_cgb = g_cgb ? 2 * g_cgb : 64; g_gb = realloc(g_gb, g_cgb * sizeof *g_gb); }
    g_gb[g_ngb++] = v;
    pthread_mutex_unlock(&g_gbmu);
}
static void garbage_free(uint64_t done) {
    pthread_mutex_lock(&g_gbmu);
    for (int i = 0; i < g_ngb;) {
        vtex *v = g_gb[i];
        if (v->dead > done || __atomic_load_n(&v->inpool, __ATOMIC_ACQUIRE) || __atomic_load_n(&v->pool_out, __ATOMIC_ACQUIRE) > done) { i++; continue; }   // the pool set may still hold its view (vk_pool)
        lk_(&g_rmu, LW_RMU);   // out of the table now (mtl_tex_release left it there)
        for (vtex **pp = &g_tex[v->serial % NHASH]; *pp; pp = &(*pp)->next) if (*pp == v) { *pp = v->next; break; }
        pthread_mutex_unlock(&g_rmu);
        for (int k = 0; k < v->nviews; k++) vkDestroyImageView(g_dev, v->views[k].v, NULL);
        if (!v->evicted) { vkDestroyImage(g_dev, v->img, NULL); img_mem_free(v->blk, v->m, v->moff, v->msize); }   // an evicted one gave its image and memory back already
        free(v->views); free(v);
        g_gb[i] = g_gb[--g_ngb];
    }
    pthread_mutex_unlock(&g_gbmu);
}
static VkFormat g_tv_fmt;   // the format of the view tex_view returned last (recording thread)
// A view of the image: type and format from the NVN view (or the texture), levels and layers, swizzle (the format's missing alpha, the texture's and the view's swizzles)
static VkImageView tex_view(vtex *v, const tex_t *t, const view_t *vw, int has_vw, int want_type, int rt) {
    int target = has_vw && vw->set_target ? vw->target : t->d.target;
    int fmtn = has_vw && vw->set_format ? vw->format : t->d.format;
    uint32_t bl = has_vw && vw->set_levels ? vw->base_level : 0, nl = has_vw && vw->set_levels ? vw->nlevels : v->levels - bl;
    uint32_t b0 = has_vw && vw->set_layers ? vw->min_layer : 0, nly = has_vw && vw->set_layers ? vw->nlayers : v->layers - b0;
    if (rt) nl = 1;   // rt 1: a render target's view; rt 2: a storage image's (one level, no swizzle, its own type); rt 3: a layered render target's (every layer from its first on)
    if (rt == 3) nly = v->layers - (b0 < v->layers ? b0 : 0);
    if (!nl || bl + nl > v->levels) nl = v->levels - (bl < v->levels ? bl : 0);
    if (!nly || b0 + nly > v->layers) nly = v->layers - (b0 < v->layers ? b0 : 0);
    fmt_t f = fmt_of(fmtn);
    if (!f.f || f.dec != v->f.dec || (f.dec && f.f != v->f.f) || fmt_block(f.f) != fmt_block(v->f.f)) f = v->f;   // a block-compressed image is viewed in its own class only
    if (v->f.depth || f.depth) f = v->f;   // a depth-stencil image is viewed in its own format (a STENCIL8 view of a D32S8 one included): the aspect picks depth or stencil
    if (f.dec) f.f = v->f.f;
    if (v->nvfl && f.f != v->vfl[0] && (v->nvfl < 2 || f.f != v->vfl[1])) {   // outside the image's format list (DM_VK_UBWC): its own format
        static int nl; if (nl++ < 8) LOG("[RED] vk: a view in format %d of an image made for %d%s (DM_VK_UBWC): viewed in the image's format", (int)f.f, (int)v->vfl[0], v->nvfl > 1 ? " and its sRGB twin" : "");
        g_ubwc_fallback++; f = v->f; }
    uint8_t swz[4] = {2, 3, 4, 5};
    if (!rt) {
        if (t->set_swz) for (int i = 0; i < 4; i++) swz[i] = (uint8_t)(t->swz >> (3 * i) & 7);
        if (has_vw && vw->set_swizzle) memcpy(swz, vw->swz, 4);
        if (f.noalpha) for (int i = 0; i < 4; i++) if (swz[i] == 5) swz[i] = 1;
    }
    VkImageViewType vt;
    if (want_type >= 0) vt = (VkImageViewType)want_type;
    else switch (target) {
        case NVN_TEXTURE_TARGET_1D: vt = VK_IMAGE_VIEW_TYPE_1D; break; case NVN_TEXTURE_TARGET_1D_ARRAY: vt = VK_IMAGE_VIEW_TYPE_1D_ARRAY; break;
        case NVN_TEXTURE_TARGET_3D: vt = VK_IMAGE_VIEW_TYPE_3D; break;
        case NVN_TEXTURE_TARGET_2D_ARRAY: case NVN_TEXTURE_TARGET_2D_MULTISAMPLE_ARRAY: vt = VK_IMAGE_VIEW_TYPE_2D_ARRAY; break;
        case NVN_TEXTURE_TARGET_CUBEMAP: vt = VK_IMAGE_VIEW_TYPE_CUBE; break; case NVN_TEXTURE_TARGET_CUBEMAP_ARRAY: vt = VK_IMAGE_VIEW_TYPE_CUBE_ARRAY; break;
        default: vt = VK_IMAGE_VIEW_TYPE_2D; break;
    }
    // A render target is a 2D view of its layer (a 2D array for several): Vulkan takes no cube or 3D view as an attachment. A cube face drawn to was a cube view of 6 layers
    // from that face on, past the image's end for faces 1-5 (Super Mario Odyssey's environment probes, its first frame in a kingdom: the Mali-G57 lost the device). A 3D
    // texture's slice is a 2D view of the image made 2D-array compatible, its slice as the layer
    if (rt == 1 || rt == 3) {
        if (target == NVN_TEXTURE_TARGET_3D) { uint32_t dd = v->d >> bl ? v->d >> bl : 1; if (b0 >= dd) b0 = 0; nly = rt == 3 ? dd - b0 : 1; vt = rt == 3 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D; }
        else if (vt == VK_IMAGE_VIEW_TYPE_CUBE || vt == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY || vt == VK_IMAGE_VIEW_TYPE_2D_ARRAY || rt == 3) vt = nly > 1 || rt == 3 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    }
    if (vt == VK_IMAGE_VIEW_TYPE_2D && nly > 1) nly = 1;
    if (vt == VK_IMAGE_VIEW_TYPE_CUBE) { nly = 6; if (b0 + 6 > v->layers) b0 = v->layers >= 6 ? v->layers - 6 : 0; }
    if (vt == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY) { nly -= nly % 6; if (!nly) { vt = VK_IMAGE_VIEW_TYPE_CUBE; nly = 6; b0 = 0; } }
    // Sampling a depth-stencil image reads depth, or its stencil where the texture or the view asks for it (DEPTH_STENCIL_MODE_STENCIL): Unreal's CopyStencilToLightingChannelsPS
    // reads the lighting channel bits from the stencil, and through the depth aspect every light lost its channels (FINAL FANTASY RESONANCE's fields drawn black)
    int want_stencil = has_vw && vw->set_dsmode ? vw->dsmode == NVN_TEXTURE_DEPTH_STENCIL_MODE_STENCIL : t->stencil;
    VkImageAspectFlags asp = v->aspect;
    if (asp & VK_IMAGE_ASPECT_DEPTH_BIT && rt != 1 && rt != 3) asp = want_stencil && (asp & VK_IMAGE_ASPECT_STENCIL_BIT) ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT;
    if (want_stencil && rt != 1 && rt != 3 && !(v->aspect & VK_IMAGE_ASPECT_STENCIL_BIT)) RED_ONCE(74, "a stencil view of texture format %#x, which has no stencil here: reads depth", fmtn);
    // The stencil where the shader looks for it: the Switch's GPU returns a stencil-mode read with the stencil in the second component (Unreal's STENCIL_COMPONENT_SWIZZLE .g on the
    // Switch), and the translated shader takes that component's raw bits; a Vulkan stencil view has it in the first, the second reads 0. Unreal's CopyStencilToLightingChannelsPS
    // ((stencil >> 4) & lighting channel bits, from .y) wrote no lighting channel anywhere, and what a channel-limited light should light stayed dark (FINAL FANTASY RESONANCE's battle
    // sprites as dark silhouettes). The second component now reads the first (an identity swizzle {R, R, B, A})
    if (asp == VK_IMAGE_ASPECT_STENCIL_BIT && !rt) for (int i = 0; i < 4; i++) if (swz[i] == 3) swz[i] = 2;
    g_tv_fmt = f.f;   // the format this view is made in (set_rt_: the attachment's format for passes and pipelines)
    uint64_t key = (uint64_t)f.f | (uint64_t)vt << 16 | (uint64_t)bl << 20 | (uint64_t)nl << 25 | (uint64_t)b0 << 30 | (uint64_t)nly << 42 | (uint64_t)(swz[0] | swz[1] << 3 | swz[2] << 6 | swz[3] << 9) << 50 | (uint64_t)(rt == 2 || rt == 3) << 62 | (uint64_t)(rt == 1 || rt == 3) << 63;
    if (asp == VK_IMAGE_ASPECT_STENCIL_BIT) key = ~key;   // the stencil aspect's views apart from the depth ones
    for (int i = 0; i < v->nviews; i++) if (v->views[i].key == key) return v->views[i].v;
    VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, v->img, vt, f.f,
                                {(VkComponentSwizzle)(swz[0] + 1), (VkComponentSwizzle)(swz[1] + 1), (VkComponentSwizzle)(swz[2] + 1), (VkComponentSwizzle)(swz[3] + 1)}, {asp, bl, nl, b0, nly}};
    VkImageView view; VkResult r = vkCreateImageView(g_dev, &vi, NULL, &view);
    if (r != VK_SUCCESS) { static int n; if (n++ < 8) LOG("[RED] vk: image view (format %d type %d levels %u+%u layers %u+%u) failed (%d)", f.f, vt, bl, nl, b0, nly, (int)r); return VK_NULL_HANDLE; }
    v->views = realloc(v->views, (v->nviews + 1) * sizeof *v->views); v->views[v->nviews++] = (vview){key, view};
    return view;
}

// ---------- Registrations (game threads) ----------
typedef struct { tex_t t; view_t v; uint8_t ok, has_v; } treg_t;
static treg_t *g_treg; static int g_ntreg;
static samp_par *g_sreg; static uint8_t *g_sreg_ok; static int g_nsreg;
void mtl_reg_tex(int id, const tex_t *t, const view_t *v) {
    if (id < 0) return;
    lk_(&g_rmu, LW_RMU);
    if (id >= g_ntreg) { int n = id + 1024; g_treg = realloc(g_treg, n * sizeof *g_treg); memset(g_treg + g_ntreg, 0, (n - g_ntreg) * sizeof *g_treg); g_ntreg = n; }
    g_treg[id].t = *t; g_treg[id].ok = 1; g_treg[id].has_v = v != NULL; if (v) g_treg[id].v = *v; atomic_fetch_add_explicit(&g_reg_gen, 1, memory_order_relaxed);
    if (g_pool_live) pool_mark(id);   // the pool set resolves this ID again at its next sync
    int dead = t->magic != MAGIC('T', 'E', 'X', 0); if (dead) g_treg_dead++;   // kept as the game registered it; tex_get never makes an image of it (the slot reads the black stand-in, as on the Metal host)
    pthread_mutex_unlock(&g_rmu);
    if (dead) { static int n; if (n++ < 4) { const uint64_t *q = (const uint64_t *)t; LOG("vk: texture ID %d registered with an object that is not a texture (finalized or reused memory; starts %016llx %016llx): its slot reads black, as on the Metal host",
                                                                                    id, (unsigned long long)q[0], (unsigned long long)q[1]); } }
}
void mtl_reg_samp(int id, const samp_par *p) {
    if (id < 0) return;
    lk_(&g_rmu, LW_RMU);
    if (id >= g_nsreg) { int n = id + 256; g_sreg = realloc(g_sreg, n * sizeof *g_sreg); g_sreg_ok = realloc(g_sreg_ok, n); memset(g_sreg_ok + g_nsreg, 0, n - g_nsreg); g_nsreg = n; }
    g_sreg[id] = *p; g_sreg_ok[id] = 1; atomic_fetch_add_explicit(&g_reg_gen, 1, memory_order_relaxed); atomic_fetch_add(&g_pool_sgen, 1);
    pthread_mutex_unlock(&g_rmu);
}
void mtl_set_tex_pool_n(int n) { atomic_store(&g_tpool_n, n); }   // the texture pools' capacity (host IDs): the pool set logs it when it is over pool_cap
void mtl_tex_init(const tex_t *t) { (void)t; }
// Released by the game (TextureFinalize): it stays findable by its serial until it is freed (garbage_free), as commands recorded before the release and replayed after it -- an
// Unreal mip-streaming copy out of the texture being replaced -- still name it. Taken out of the table at once, such a copy found no image, made a new one from the texture's
// storage, which lacks what GPU copies put into the image, and copied black into the new texture's lower levels (FINAL FANTASY RESONANCE's corridor: dark columns at a distance)
void mtl_tex_release(tex_t *t) {
    lk_(&g_rmu, LW_RMU);
    vtex *v = tex_find(t->serial); int fresh = v && !v->dead;
    pthread_mutex_unlock(&g_rmu);
    if (fresh) tex_free(v);
}
void mtl_tex_cpu_written(const tex_t *t) {
    lk_(&g_rmu, LW_RMU);
    vtex *v = tex_find(t->serial); if (v) v->dirty = 1;
    if (v && v->inpool) atomic_fetch_add(&g_pool_cpu, 1);   // the pool set uploads it again at its next sync
    if (!cpu_written(t->serial)) { if (g_ncpu == g_ccpu) { g_ccpu = g_ccpu ? 2 * g_ccpu : 64; g_cpu_written = realloc(g_cpu_written, g_ccpu * sizeof *g_cpu_written); } g_cpu_written[g_ncpu++] = t->serial; }
    pthread_mutex_unlock(&g_rmu);
}
void mtl_prog_release(prog_t *p) { (void)p; }

// Samplers, cached by their parameters
typedef struct { samp_par p; VkSampler s; } scache;
static scache *g_sc; static int g_nsc;
static VkSamplerAddressMode wrap(int w) {
    switch (w) { case 1: return VK_SAMPLER_ADDRESS_MODE_REPEAT; case 6: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT; case 4: case 5: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
                 case 2: case 3: return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE; default: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; }
}
static VkSampler samp_get(int sid) {
    if (sid <= 0 || sid >= g_nsreg || !g_sreg_ok[sid]) return g_dummy_samp;
    samp_par p = g_sreg[sid];
    static struct { int sid; samp_par p; VkSampler s; } dm[1024];   // by sampler ID, checked against its parameters (an ID can be registered again with others)
    if (dm[sid % 1024].sid == sid && dm[sid % 1024].s && !memcmp(&dm[sid % 1024].p, &p, sizeof p)) return dm[sid % 1024].s;
    for (int i = 0; i < g_nsc; i++) if (!memcmp(&g_sc[i].p, &p, sizeof p)) { dm[sid % 1024].sid = sid; dm[sid % 1024].p = p; dm[sid % 1024].s = g_sc[i].s; return g_sc[i].s; }
    VkSamplerCreateInfo si = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = p.magf ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    si.minFilter = p.minf == 1 || p.minf == 3 || p.minf == 5 ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    si.mipmapMode = p.minf == 4 || p.minf == 5 ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = wrap(p.ws); si.addressModeV = wrap(p.wt); si.addressModeW = wrap(p.wr);
    si.mipLodBias = p.lod_bias; si.minLod = p.minf >= 2 ? p.lod_min : 0; si.maxLod = p.minf >= 2 ? p.lod_max : 0.25f;
    if (p.aniso > 1) { si.anisotropyEnable = VK_TRUE; si.maxAnisotropy = p.aniso > 16 ? 16 : p.aniso; }
    if (p.cmp_mode) { si.compareEnable = VK_TRUE; si.compareOp = (VkCompareOp)(p.cmp_func ? p.cmp_func - 1 : 7); }
    int white = p.border[0] + p.border[1] + p.border[2] > 1.5f, opaque = p.border[3] > 0.5f;
    si.borderColor = white ? VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE : opaque ? VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK : VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    VkSampler s; if (vkCreateSampler(g_dev, &si, NULL, &s) != VK_SUCCESS) return g_dummy_samp;
    g_sc = realloc(g_sc, (g_nsc + 1) * sizeof *g_sc); g_sc[g_nsc++] = (scache){p, s};
    return s;
}

// ---------- Shaders and pipelines ----------
enum { BT_UBO, BT_SSBO, BT_CIS, BT_TBUF, BT_UTB, BT_STB, BT_SIMG, BT_TEX, BT_SAMP };
enum { K_UNK, K_CBUF, K_C1, K_SUPPORT, K_DRVCB, K_TEX, K_IMG, K_SBUF, K_TBUF, K_UBOP, K_TEXU, K_TPOOL, K_SPOOL };   // K_TPOOL / K_SPOOL: the texture / sampler pool indexed by ID (vk_pool)   // K_UBOP: a compute uniform buffer read through its address (dmsh_build.py ubo_ptr), bound as storage;
                                                                                                // K_TEXU: a bindless texture handle in uniform slot reg at byte off (tex_ubo)
enum { S_NONE, S_SAMP, S_SAMP_TEX, S_SAMP_ID, S_SAMP_UBO };   // S_SAMP_UBO: the sampler's handle in uniform slot sreg at byte soff (samp_ubo)
// off / soff: byte offsets in a uniform buffer, up to its 64 KB -- 32-bit (vk-t11): as int16_t, Tomodachi Life's island composite read its handles at cbuf 0 + 40928..40952 as
// -24608..-24584, before the buffer (zero, the stand-in bound: the beach and inlets black)
typedef struct { uint32_t binding, count; uint8_t type, kind, skind; int16_t reg, sreg; int32_t off, soff; } sbind;
typedef struct {
    char key[17]; int stage;   // 0 vertex, 1 fragment, 5 compute (NVN stage numbers)
    VkShaderModule mod, mod_dual; VkDescriptorSetLayout dsl;   // mod_dual: a fragment shader's output 1 as the second source of dual-source blending (spv_dual)
    int dual_tried;
    sbind b[64]; int nb; uint8_t vin[32]; uint32_t vin_mask;   // vin: component type per vertex input location (0 float, 1 uint, 2 int)
    uint8_t *c1; uint32_t c1_len; uint32_t tg[3];
    VkPipelineLayout cpl; VkPipeline cpp;   // compute shaders: their layout and pipeline (dispatch)
    uint8_t dynu[64];   // uniform buffer bindings made dynamic (UNIFORM_BUFFER_DYNAMIC): their ring offset goes with the bind, so a set stays the same while only uniform data changes
    int layer_out;   // writes the Layer builtin (a vertex or geometry shader that picks the attachment layer per primitive): its draws run in a layered pass
    int vid;   // a vertex shader that reads VertexIndex, patched to add the push constant at offset 0 (spv_vid_bias): its layout has the range, draw() pushes vlo
    uint8_t md[8];   // the key as bytes (shader_get's lookup)
    int dyn;   // dynamic slot arrays (profile key tex_dyn_slots): 32-entry texture / texel buffer / sampler arrays indexed at run time by the IDs the shader reads from c2 (see write_set)
    int ssbo_wr;   // the module stores to or atomically updates a storage buffer (spv_ssbo_writes): only then is a draw's storage buffer a write that gets lost
    uint32_t io_mask;   // user interface locations: a vertex shader's outputs, a pixel shader's inputs (spv_io): an input no output feeds is an interface mismatch (vs_pad_mod)
    uint16_t in_type[32];   // a pixel shader's input type per location (spv_io: 1-4 components | 0x100 int | 0x200 uint; 0 = not a plain 32-bit scalar or vector)
    struct { uint32_t mask; VkShaderModule m; } pad[8]; int npad, pad_full;   // a vertex shader's padded variants, by the locations added (vs_pad_mod)
    int io_complex;   // a user interface variable that is not a scalar or vector (an array, matrix or struct takes several locations): never padded
    int pool;   // its texture / sampler pools were moved into the pool set (pool_shader): its layout has set SPV_POOL_SET, draws bind pool_set() there
} shader_t;
static shader_t **g_sh; static int g_nsh;
static int g_depth_zero;   // NVN depth mode (mtl_set_depth_zero, defined with the replay state)
static const char *g_vkdir;
static uint32_t *read_file(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint32_t *p = malloc((size_t)sz + 1); if (fread(p, 1, (size_t)sz, f) != (size_t)sz) { free(p); fclose(f); return NULL; }
    fclose(f); ((char *)p)[sz] = 0; *n = (size_t)sz; return p;
}
// NVN's GL depth convention (nvnDeviceSetDepthMode near = -w: Metroid Dread, Super Mario Odyssey...): clip-space z runs -w..w, Vulkan's 0..w. A vertex shader then ends with
// z' = (z + w) / 2 -- the GL entry point dmsh_build.py adds to the Metal libraries (vs_depth), done here on the SPIR-V: before every return of the entry point the position
// (a BuiltIn Position variable, or member of an output block) is read, rewritten and stored back. Returns a new word array (the caller frees it), NULL when no position is written
// Position is decorated Invariant in every vertex shader: games draw a depth prepass with one vertex shader and the scene with another under the depth test EQUAL (Metroid Dread's
// G-buffer pass), which on the Switch matches bit for bit; without the decoration the Mali compiler evaluated the two positions differently and the whole level failed the test
static uint32_t *spv_invariant(const uint32_t *w, size_t nw, size_t *out_n) {
    enum { OpDecorate = 71, OpMemberDecorate = 72, BuiltIn = 11, Position = 0, Invariant = 18 };
    for (size_t i = 5; i < nw;) {   // already invariant: nothing to add
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16; if (!len) return NULL;
        if ((op == OpDecorate && len >= 3 && w[i + 2] == Invariant) || (op == OpMemberDecorate && len >= 4 && w[i + 3] == Invariant)) return NULL;
        i += len;
    }
    uint32_t *o = malloc((nw + 8) * 4); size_t n = 0; int added = 0;
    memcpy(o, w, 20); n = 5;
    for (size_t i = 5; i < nw;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        memcpy(o + n, w + i, len * 4); n += len;
        if (added < 2 && op == OpMemberDecorate && len >= 5 && w[i + 3] == BuiltIn && w[i + 4] == Position) { o[n++] = 4u << 16 | OpMemberDecorate; o[n++] = w[i + 1]; o[n++] = w[i + 2]; o[n++] = Invariant; added++; }
        else if (added < 2 && op == OpDecorate && len >= 4 && w[i + 2] == BuiltIn && w[i + 3] == Position) { o[n++] = 3u << 16 | OpDecorate; o[n++] = w[i + 1]; o[n++] = Invariant; added++; }
        i += len;
    }
    if (!added) { free(o); return NULL; }
    *out_n = n; return o;
}
// PointSize = 1.0 stored first thing in every vertex shader: Vulkan leaves a point's size undefined unless the vertex shader writes it, and the translated shaders write it only when
// the Switch shader did (NVN then takes SetPointSize). Super Mario Odyssey draws points with shaders that do not, and the Mali-G57 lost the device at its first such draw. A shader that
// writes its own size later overrides the 1.0; topologies other than points ignore it
// Dual-source blending: NVN's pixel shader writes the second source as its output 1, and a blend factor SRC1_* reads it; Vulkan reads the second source from the output at
// location 0, index 1 (as Metal's [[color(0), index(1)]], dmsh_build.py fs_dual). A copy of the module whose location-1 output variable is moved there (its Location
// decoration made 0, an Index 1 decoration added); NULL when the shader has no location-1 output
static uint32_t *spv_dual(const uint32_t *w, size_t nw, size_t *out_n) {
    enum { OpVariable = 59, OpDecorate = 71, Output = 3, Location = 30, Index = 32 };
    size_t at = 0;
    for (size_t i = 5; i < nw && !at;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16; if (!len) return NULL;
        if (op == OpDecorate && len >= 4 && w[i + 2] == Location && w[i + 3] == 1)
            for (size_t j = 5; j < nw;) { uint32_t o2 = w[j] & 0xffff, l2 = w[j] >> 16; if (!l2) return NULL; if (o2 == OpVariable && l2 >= 4 && w[j + 2] == w[i + 1] && w[j + 3] == Output) { at = i; break; } j += l2; }
        i += len;
    }
    if (!at) return NULL;
    uint32_t *o = malloc((nw + 4) * 4); if (!o) return NULL;
    memcpy(o, w, (at + 4) * 4); o[at + 3] = 0;
    o[at + 4] = 4u << 16 | OpDecorate; o[at + 5] = w[at + 1]; o[at + 6] = Index; o[at + 7] = 1;
    memcpy(o + at + 8, w + at + 4, (nw - at - 4) * 4);
    *out_n = nw + 4; return o;
}
// The user locations of a module's interface: Output variables of a vertex shader or Input variables of a pixel shader (built-ins left out), and for a pixel shader each input's
// type when it is a 32-bit float / int / uint scalar or vector with no Component decoration (in_type; 0 otherwise)
static uint32_t spv_io(const uint32_t *w, size_t nw, int sc, uint16_t *types, int *complex) {
    enum { OpTypeInt = 21, OpTypeFloat = 22, OpTypeVector = 23, OpTypePointer = 32, OpVariable = 59, OpDecorate = 71, OpMemberDecorate = 72, Location = 30, Component = 31, BuiltIn = 11 };
    uint32_t bound = nw > 3 ? w[3] : 0, mask = 0; if (!bound || bound > (1u << 22)) return 0;
    int32_t *loc = malloc(bound * 4); uint8_t *bi = calloc(bound, 1), *comp = calloc(bound, 1); uint32_t *tdef = calloc(bound, 4);
    if (!loc || !bi || !comp || !tdef) { free(loc); free(bi); free(comp); free(tdef); return 0; }
    for (uint32_t i = 0; i < bound; i++) loc[i] = -1;
    for (size_t i = 5; i < nw;) { uint32_t op = w[i] & 0xffff, len = w[i] >> 16; if (!len || i + len > nw) break;
        if (op == OpDecorate && len >= 4 && w[i + 1] < bound) { if (w[i + 2] == Location) loc[w[i + 1]] = (int32_t)w[i + 3]; else if (w[i + 2] == BuiltIn) bi[w[i + 1]] = 1; else if (w[i + 2] == Component) comp[w[i + 1]] = 1; }
        if (op == OpMemberDecorate && len >= 4 && w[i + 1] < bound && w[i + 3] == BuiltIn) bi[w[i + 1]] = 1;
        if (op == OpTypeFloat && len >= 3 && w[i + 1] < bound && w[i + 2] == 32) tdef[w[i + 1]] = 1;
        if (op == OpTypeInt && len >= 4 && w[i + 1] < bound && w[i + 2] == 32) tdef[w[i + 1]] = w[i + 3] ? 0x101 : 0x201;
        if (op == OpTypeVector && len >= 4 && w[i + 1] < bound && w[i + 2] < bound && tdef[w[i + 2]] && w[i + 3] >= 2 && w[i + 3] <= 4) tdef[w[i + 1]] = (tdef[w[i + 2]] & 0xf00) | w[i + 3];
        if (op == OpTypePointer && len >= 4 && w[i + 1] < bound && w[i + 2] == (uint32_t)sc && w[i + 3] < bound) tdef[w[i + 1]] = 0x10000 | w[i + 3];   // a pointer: its pointee
        if (op == OpVariable && len >= 4 && w[i + 3] == (uint32_t)sc && w[i + 2] < bound && w[i + 1] < bound && !bi[w[i + 2]] && loc[w[i + 2]] >= 0 && loc[w[i + 2]] < 32) {
            mask |= 1u << loc[w[i + 2]];
            { uint32_t pt = tdef[w[i + 1]]; if (!(pt >> 16) || !tdef[pt & 0xffff]) *complex = 1; }   // an array / matrix / struct covers more locations than its bit: the mask is not exact
            if (types) { uint32_t pt = tdef[w[i + 1]], ty = pt >> 16 ? tdef[pt & 0xffff] : 0; types[loc[w[i + 2]]] = comp[w[i + 2]] || ty > 0x2ff ? 0 : (uint16_t)ty; }
        }
        i += len; }
    free(loc); free(bi); free(comp); free(tdef);
    return mask;
}
// A vertex shader with an Output at every location in add, of the pixel shader's input type there (types), written with zeros at the entry function's start: a pixel shader
// reading an input the vertex shader lacks is an interface mismatch Vulkan leaves undefined (VUID-RuntimeSpirv-OpEntryPoint-08743), which the Adreno 825 refuses (-13).
// Tomodachi Life: Living the Dream's c0b81b887fb0137d is the vertex stage of a tessellation program, drawn without tessellation as on Metal; its pixel shader d317a98e994ca9dd
// reads location 2, which the evaluation stage would write. Metal's paired-vertex-shader padding gives zeros there too. Existing types are used; a missing vector or pointer type
// and the null constants are added before the first function. NULL when a location's type is not a plain 32-bit scalar or vector, or the module's layout is not as expected
static uint32_t *spv_pad_outputs(const uint32_t *w, size_t nw, uint32_t add, const uint16_t *types, size_t *out_n) {
    enum { OpEntryPoint = 15, OpTypeInt = 21, OpTypeFloat = 22, OpTypeVector = 23, OpTypePointer = 32, OpConstantNull = 46, OpFunction = 54, OpLabel = 248, OpVariable = 59,
           OpStore = 62, OpDecorate = 71, Output = 3, Location = 30, Function = 7 };
    for (int l = 0; l < 32; l++) if (add >> l & 1 && !types[l]) return NULL;
    uint32_t bound = w[3], tf = 0, ti = 0, tu = 0, tv[3][5] = {{0}}, tp[3][5] = {{0}}, tn[3][5] = {{0}}, entry = 0;
    size_t first_type = 0, first_fn = 0, ep = 0, body = 0;
    for (size_t i = 5; i < nw;) { uint32_t op = w[i] & 0xffff, len = w[i] >> 16; if (!len || i + len > nw) return NULL;
        if (op == OpEntryPoint && len >= 3 && w[i + 1] == 0 && !ep) { ep = i; entry = w[i + 2]; }
        if (!first_type && ((op >= 19 && op <= 39) || (op >= 41 && op <= 52) || op == OpVariable)) first_type = i;
        if (op == OpTypeFloat && len >= 3 && w[i + 2] == 32) tf = w[i + 1];
        if (op == OpTypeInt && len >= 4 && w[i + 2] == 32) { if (w[i + 3]) ti = w[i + 1]; else tu = w[i + 1]; }
        for (int k = 0; k < 3; k++) { uint32_t base = k == 0 ? tf : k == 1 ? ti : tu; if (!base) continue;
            if (op == OpTypeVector && len >= 4 && w[i + 2] == base && w[i + 3] >= 2 && w[i + 3] <= 4) tv[k][w[i + 3]] = w[i + 1];
            for (int c = 1; c <= 4; c++) { uint32_t t = c == 1 ? base : tv[k][c]; if (!t) continue;
                if (op == OpTypePointer && len >= 4 && w[i + 2] == Output && w[i + 3] == t) tp[k][c] = w[i + 1];
                if (op == OpConstantNull && len >= 3 && w[i + 1] == t) tn[k][c] = w[i + 2]; } }
        if (op == OpFunction && !first_fn) first_fn = i;
        if (op == OpFunction && len >= 5 && w[i + 2] == entry && !body) {   // the entry function: after its first label and the OpVariable block
            size_t j = i + len; if (j < nw && (w[j] & 0xffff) == OpLabel) { j += w[j] >> 16; while (j < nw && (w[j] & 0xffff) == OpVariable && (w[j] >> 16) >= 4 && w[j + 3] == Function) j += w[j] >> 16; body = j; } }
        i += len; }
    if (!ep || !first_type || !first_fn || !body) return NULL;
    uint32_t *add_dec = malloc(32 * 4 * 4), *add_ty = malloc(32 * 4 * 16), *add_st = malloc(32 * 3 * 4), *add_ep = malloc(32 * 4);
    size_t nd = 0, nt = 0, ns = 0, ne = 0;
    if (!add_dec || !add_ty || !add_st || !add_ep) { free(add_dec); free(add_ty); free(add_st); free(add_ep); return NULL; }
    for (int l = 0; l < 32; l++) { if (!(add >> l & 1)) continue;
        int k = types[l] & 0x100 ? 1 : types[l] & 0x200 ? 2 : 0, c = types[l] & 0xff; uint32_t *base = k == 0 ? &tf : k == 1 ? &ti : &tu;
        if (!*base) { *base = bound++; if (k == 0) { add_ty[nt++] = 3u << 16 | OpTypeFloat; add_ty[nt++] = *base; add_ty[nt++] = 32; }
                                     else { add_ty[nt++] = 4u << 16 | OpTypeInt; add_ty[nt++] = *base; add_ty[nt++] = 32; add_ty[nt++] = k == 1; } }
        uint32_t t = *base;
        if (c > 1) { if (!tv[k][c]) { tv[k][c] = bound++; add_ty[nt++] = 4u << 16 | OpTypeVector; add_ty[nt++] = tv[k][c]; add_ty[nt++] = *base; add_ty[nt++] = (uint32_t)c; } t = tv[k][c]; }
        if (!tp[k][c]) { tp[k][c] = bound++; add_ty[nt++] = 4u << 16 | OpTypePointer; add_ty[nt++] = tp[k][c]; add_ty[nt++] = Output; add_ty[nt++] = t; }
        if (!tn[k][c]) { tn[k][c] = bound++; add_ty[nt++] = 3u << 16 | OpConstantNull; add_ty[nt++] = t; add_ty[nt++] = tn[k][c]; }
        uint32_t v = bound++;
        add_ty[nt++] = 4u << 16 | OpVariable; add_ty[nt++] = tp[k][c]; add_ty[nt++] = v; add_ty[nt++] = Output;
        add_dec[nd++] = 4u << 16 | OpDecorate; add_dec[nd++] = v; add_dec[nd++] = Location; add_dec[nd++] = (uint32_t)l;
        add_st[ns++] = 3u << 16 | OpStore; add_st[ns++] = v; add_st[ns++] = tn[k][c];
        add_ep[ne++] = v; }
    uint32_t *o = malloc((nw + nd + nt + ns + ne) * 4); size_t n = 0;
    if (o) {
        uint32_t eplen = w[ep] >> 16;
        for (size_t i = 0; i < nw;) {
            if (i == first_type) { memcpy(o + n, add_dec, nd * 4); n += nd; }
            if (i == first_fn) { memcpy(o + n, add_ty, nt * 4); n += nt; }
            if (i == body) { memcpy(o + n, add_st, ns * 4); n += ns; }
            if (i < 5) { o[n++] = w[i++]; continue; }
            uint32_t len = w[i] >> 16;
            if (i == ep) { o[n] = (uint32_t)(eplen + ne) << 16 | OpEntryPoint; memcpy(o + n + 1, w + i + 1, (eplen - 1) * 4); memcpy(o + n + eplen, add_ep, ne * 4); n += eplen + ne; }
            else { memcpy(o + n, w + i, len * 4); n += len; }
            i += len;
        }
        o[3] = bound; *out_n = n;
    }
    free(add_dec); free(add_ty); free(add_st); free(add_ep);
    return o;
}
static uint32_t *vs_transform(shader_t *s, uint32_t *spv, size_t *ns, int at_load);
static pthread_mutex_t g_padmu = PTHREAD_MUTEX_INITIALIZER;
// The vertex shader to draw with a pixel shader: itself when it writes every location the pixel shader reads (every pair in FINAL FANTASY RESONANCE, FINAL FANTASY XII and Harvest
// Moon), else a variant padded with zero outputs at the missing ones (spv_pad_outputs), made from the shader file with the same transforms as the plain one, cached by the mask
static VkShaderModule vs_pad_mod(shader_t *vs, const shader_t *fs) {
    uint32_t miss = fs ? fs->io_mask & ~vs->io_mask : 0; if (!miss || vs->io_complex) return vs->mod;   // a vertex shader with an array / matrix / struct output: its mask is not exact, never padded
    pthread_mutex_lock(&g_padmu);
    VkShaderModule m = VK_NULL_HANDLE; int found = 0;
    for (int i = 0; i < vs->npad; i++) if (vs->pad[i].mask == miss) { m = vs->pad[i].m; found = 1; break; }
    if (!found && vs->npad == 8) {   // full: drawn unpadded, as before (logged once per shader)
        if (!vs->pad_full) { vs->pad_full = 1; LOG("vk: vertex shader %s: more than 8 padded variants; pixel shader %s drawn with the plain one", vs->key, fs->key); }
        pthread_mutex_unlock(&g_padmu); return vs->mod;
    }
    if (!found) {
        char path[1024]; size_t ns = 0, n = 0; snprintf(path, sizeof path, "%s/%s.spv", g_vkdir, vs->key);
        uint32_t *spv = read_file(path, &ns), *w = spv ? spv_pad_outputs(spv, ns / 4, miss, fs->in_type, &n) : NULL; free(spv);
        if (w) { size_t nb = n * 4; w = vs_transform(vs, w, &nb, 0);
                 VkShaderModuleCreateInfo mi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, nb, w};
                 if (vkCreateShaderModule(g_dev, &mi, NULL, &m) != VK_SUCCESS) m = VK_NULL_HANDLE; free(w); }
        LOG("vk: vertex shader %s padded with zero outputs at locations %#x for pixel shader %s (it reads inputs the vertex shader does not write): %s", vs->key, miss, fs->key, m ? "made" : "not possible, drawn unpadded");
        vs->pad[vs->npad].mask = miss; vs->pad[vs->npad].m = m; vs->npad++;
    }
    pthread_mutex_unlock(&g_padmu);
    return m ? m : vs->mod;
}
static void spv_set(uint32_t *spv, size_t nw, uint32_t set) {   // every resource into descriptor set `set` (DescriptorSet decorations, operand 34): the fragment stage's go to set 1
    for (size_t i = 5; i < nw;) { uint32_t w = spv[i], op = w & 0xffff, len = w >> 16; if (!len) break; if (op == 71 && len >= 4 && spv[i + 2] == 34) spv[i + 3] = set; i += len; }
}
static VkShaderModule fs_dual_mod_(shader_t *fs);
static pthread_mutex_t g_dualmu = PTHREAD_MUTEX_INITIALIZER;
static VkShaderModule fs_dual_mod(shader_t *fs) { pthread_mutex_lock(&g_dualmu); VkShaderModule m = fs_dual_mod_(fs); pthread_mutex_unlock(&g_dualmu); return m; }
static VkShaderModule fs_dual_mod_(shader_t *fs) {   // made at the first pipeline that blends with the second source, from the shader file read again
    if (fs->mod_dual || fs->dual_tried) return fs->mod_dual;
    fs->dual_tried = 1;
    char path[1024]; size_t ns = 0, n; snprintf(path, sizeof path, "%s/%s.spv", g_vkdir, fs->key);
    uint32_t *spv = read_file(path, &ns), *w = spv ? spv_dual(spv, ns / 4, &n) : NULL; free(spv);
    if (!w) { LOG("[RED] vk: pixel shader %s: a pipeline blends with the second source, but the shader writes no output 1", fs->key); return VK_NULL_HANDLE; }
    spv_set(w, n, 1);
    VkShaderModuleCreateInfo mi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, n * 4, w};
    if (vkCreateShaderModule(g_dev, &mi, NULL, &fs->mod_dual) != VK_SUCCESS) { LOG("[RED] vk: pixel shader %s: its dual-source module was rejected", fs->key); fs->mod_dual = VK_NULL_HANDLE; }
    else LOG("vk: pixel shader %s: output 1 is the second source of dual-source blending", fs->key);
    free(w);
    return fs->mod_dual;
}
static uint32_t *spv_point_size(const uint32_t *w, size_t nw, size_t *out_n) {
    enum { OpEntryPoint = 15, OpTypeInt = 21, OpTypeFloat = 22, OpTypePointer = 32, OpConstant = 43, OpFunction = 54, OpLabel = 248, OpVariable = 59, OpStore = 62, OpAccessChain = 65,
           OpDecorate = 71, OpMemberDecorate = 72, Output = 3, BuiltIn = 11, PointSize = 1 };
    uint32_t bound = w[3], entry = 0, f32 = 0, u32 = 0, ps_struct = 0, ps_member = 0, ps_var = 0, var = 0; int member_path = 0; size_t first_fn = 0;
    for (size_t i = 5; i < nw;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16; if (!len) return NULL;
        if (op == OpEntryPoint && !entry) entry = w[i + 2];
        else if (op == OpMemberDecorate && len >= 5 && w[i + 3] == BuiltIn && w[i + 4] == PointSize) { ps_struct = w[i + 1]; ps_member = w[i + 2]; member_path = 1; }
        else if (op == OpDecorate && len >= 4 && w[i + 2] == BuiltIn && w[i + 3] == PointSize) ps_var = w[i + 1];
        else if (op == OpTypeFloat && w[i + 2] == 32) f32 = w[i + 1];
        else if (op == OpTypeInt && w[i + 2] == 32 && !w[i + 3] && !u32) u32 = w[i + 1];
        else if (op == OpFunction && !first_fn) first_fn = i;
        i += len;
    }
    if (!entry || !f32 || !first_fn || (!member_path && !ps_var) || (member_path && !u32)) return NULL;
    for (size_t i = 5; i < first_fn;) {   // the Output variable holding PointSize (the per-vertex block, or the variable itself)
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (op == OpVariable && w[i + 3] == Output) {
            if (!member_path && w[i + 2] == ps_var) var = ps_var;
            if (member_path) for (size_t j = 5; j < first_fn;) { uint32_t o2 = w[j] & 0xffff, l2 = w[j] >> 16; if (o2 == OpTypePointer && w[j + 1] == w[i + 1] && w[j + 3] == ps_struct) var = w[i + 2]; j += l2; }
        }
        i += len;
    }
    if (!var) return NULL;
    // the entry function's first block: after its OpLabel and OpVariables
    size_t at = 0;
    for (size_t i = first_fn; i < nw;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (op == OpFunction && w[i + 2] != entry) { i += len; continue; }
        if (op == OpFunction) { size_t j = i + len; while (j < nw && (w[j] & 0xffff) != OpLabel) j += w[j] >> 16; j += w[j] >> 16; while (j < nw && (w[j] & 0xffff) == OpVariable) j += w[j] >> 16; at = j; break; }
        i += len;
    }
    if (!at) return NULL;
    uint32_t ptr = 0, one = bound++, cidx = bound++, ac = bound++;   // the Output float pointer type is reused when declared (a second declaration of it is invalid)
    for (size_t i = 5; i < first_fn;) { uint32_t op = w[i] & 0xffff, len = w[i] >> 16; if (op == OpTypePointer && w[i + 2] == Output && w[i + 3] == f32) ptr = w[i + 1]; i += len; }
    int new_ptr = !ptr; if (new_ptr) ptr = bound++;
    uint32_t *o = malloc((nw + 32) * 4); size_t n = 0;
    memcpy(o, w, first_fn * 4); n = first_fn;
    if (new_ptr) { o[n++] = 4u << 16 | OpTypePointer; o[n++] = ptr; o[n++] = Output; o[n++] = f32; }
    o[n++] = 4u << 16 | OpConstant; o[n++] = f32; o[n++] = one; o[n++] = 0x3f800000;
    if (member_path) { o[n++] = 4u << 16 | OpConstant; o[n++] = u32; o[n++] = cidx; o[n++] = ps_member; }
    memcpy(o + n, w + first_fn, (at - first_fn) * 4); n += at - first_fn;
    if (member_path) { o[n++] = 5u << 16 | OpAccessChain; o[n++] = ptr; o[n++] = ac; o[n++] = var; o[n++] = cidx; o[n++] = 3u << 16 | OpStore; o[n++] = ac; o[n++] = one; }
    else { o[n++] = 3u << 16 | OpStore; o[n++] = var; o[n++] = one; }
    memcpy(o + n, w + at, (nw - at) * 4); n += nw - at;
    o[3] = bound; *out_n = n; return o;
}
// The vertex index a shader reads, as the game gave it. draw() copies only the vertices a draw uses (from vertex vlo on) and shifts the draw's vertex numbers down by vlo,
// so VertexIndex came out vlo too low. Unreal's vertex shaders here fetch everything but the position from texel buffers by VertexIndex (manual vertex fetch): every mesh
// section whose indices do not start at 0 took another vertex's UVs and tangents (FINAL FANTASY RESONANCE's castle: the upper walls dark, smeared and lit as if facing down,
// the grand staircase dark and jagged; the Metal host binds the pool in place, no shift). Every load of the VertexIndex variable gets a 4-byte push constant (offset 0,
// vertex stage) added, which draw() sets to vlo. Returns a new word array (the caller frees it), NULL when the shader does not read VertexIndex
// DM_VK_VTXIDX_FIX=0 leaves the shaders as they were (VertexIndex vlo too low). g_vid_push / g_vid_nz: draws that pushed the bias, and those whose bias was not 0 (the recording line)
static uint64_t g_vid_push, g_vid_nz;
static int vid_fix_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_VTXIDX_FIX"); v = !(e && !strcmp(e, "0")); if (!v) LOG("DM_VK_VTXIDX_FIX=0: vertex shaders read VertexIndex shifted by the draw's first vertex copied"); } return v; }
static uint32_t *spv_vid_bias(const uint32_t *w, size_t nw, size_t *out_n) {
    enum { OpEntryPoint = 15, OpTypeInt = 21, OpTypeStruct = 30, OpTypePointer = 32, OpConstant = 43, OpFunction = 54, OpVariable = 59, OpLoad = 61, OpAccessChain = 65,
           OpDecorate = 71, OpMemberDecorate = 72, OpName = 5, OpIAdd = 128, Input = 1, PushConstant = 9, BuiltIn = 11, VertexIndex = 42, Block = 2, Offset = 35 };
    uint32_t bound = w[3], var = 0, ptr = 0, ty = 0; size_t first_fn = 0, dec_at = 0, ep_at = 0; int nload = 0;
    for (size_t i = 5; i < nw;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16; if (!len || i + len > nw) return NULL;
        if (op == OpDecorate && len >= 4 && w[i + 2] == BuiltIn && w[i + 3] == VertexIndex) { if (var) return NULL; var = w[i + 1]; dec_at = i + len; }
        else if (op == OpMemberDecorate && len >= 5 && w[i + 3] == BuiltIn && w[i + 4] == VertexIndex) return NULL;   // in an input block: not handled (not seen)
        else if (op == OpEntryPoint && !ep_at) ep_at = i;
        else if (op == OpFunction && !first_fn) first_fn = i;
        i += len;
    }
    if (!var || !first_fn || !dec_at || !ep_at) return NULL;
    for (size_t i = 5; i < nw;) {   // the variable's pointer type and the integer it holds; the translator only ever loads it
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (op == OpVariable && len >= 4 && w[i + 2] == var && w[i + 3] == Input) ptr = w[i + 1];
        else if (op == OpLoad && len >= 4 && w[i + 3] == var) nload++;
        i += len;
    }
    for (size_t i = 5; i < first_fn && ptr;) { uint32_t op = w[i] & 0xffff, len = w[i] >> 16; if (op == OpTypePointer && w[i + 1] == ptr) ty = w[i + 3]; i += len; }
    int is_int = 0; for (size_t i = 5; i < first_fn && ty;) { uint32_t op = w[i] & 0xffff, len = w[i] >> 16; if (op == OpTypeInt && w[i + 1] == ty && w[i + 2] == 32) is_int = 1; i += len; }
    if (!ptr || !is_int || !nload) return NULL;
    uint32_t st = bound++, pst = bound++, pvar = bound++, pty = bound++, cz = bound++;
    int add_if = w[1] >= 0x10400;   // SPIR-V 1.4 on: the entry point lists every global variable it uses
    uint32_t *o = malloc((nw + 40 + (size_t)nload * 13) * 4); if (!o) return NULL; size_t n = 0;
    for (size_t i = 0; i < 5; i++) o[n++] = w[i];
    for (size_t i = 5; i < nw;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (i == first_fn) {
            o[n++] = 3u << 16 | OpTypeStruct; o[n++] = st; o[n++] = ty;
            o[n++] = 4u << 16 | OpTypePointer; o[n++] = pst; o[n++] = PushConstant; o[n++] = st;
            o[n++] = 4u << 16 | OpVariable; o[n++] = pst; o[n++] = pvar; o[n++] = PushConstant;
            o[n++] = 4u << 16 | OpTypePointer; o[n++] = pty; o[n++] = PushConstant; o[n++] = ty;
            o[n++] = 4u << 16 | OpConstant; o[n++] = ty; o[n++] = cz; o[n++] = 0;
        }
        if (i == dec_at) { o[n++] = 3u << 16 | OpDecorate; o[n++] = st; o[n++] = Block; o[n++] = 5u << 16 | OpMemberDecorate; o[n++] = st; o[n++] = 0; o[n++] = Offset; o[n++] = 0; }
        if (op == OpEntryPoint && i == ep_at && add_if) { o[n++] = (len + 1) << 16 | op; memcpy(o + n, w + i + 1, (len - 1) * 4); n += len - 1; o[n++] = pvar; i += len; continue; }
        if (op == OpLoad && len >= 4 && w[i + 3] == var) {   // r = load(var) becomes t = load(var); b = load(push constant); r = t + b
            uint32_t r = w[i + 2], t = bound++, ac = bound++, b = bound++;
            memcpy(o + n, w + i, len * 4); o[n + 2] = t; n += len;
            o[n++] = 5u << 16 | OpAccessChain; o[n++] = pty; o[n++] = ac; o[n++] = pvar; o[n++] = cz;
            o[n++] = 4u << 16 | OpLoad; o[n++] = ty; o[n++] = b; o[n++] = ac;
            o[n++] = 5u << 16 | OpIAdd; o[n++] = ty; o[n++] = r; o[n++] = t; o[n++] = b;
            i += len; continue;
        }
        memcpy(o + n, w + i, len * 4); n += len; i += len;
    }
    o[3] = bound; *out_n = n; return o;
}
static uint32_t *spv_gl_depth(const uint32_t *w, size_t nw, size_t *out_n) {
    enum { OpEntryPoint = 15, OpTypeInt = 21, OpTypeFloat = 22, OpTypeVector = 23, OpTypePointer = 32, OpConstant = 43, OpFunction = 54, OpVariable = 59, OpLoad = 61, OpStore = 62,
           OpAccessChain = 65, OpDecorate = 71, OpMemberDecorate = 72, OpCompositeExtract = 81, OpCompositeInsert = 82, OpFAdd = 129, OpFMul = 133, OpReturn = 253, Output = 3 };
    uint32_t bound = w[3], entry = 0, f32 = 0, v4 = 0, u32 = 0, pos_struct = 0, pos_member = 0, pos_var = 0, ptr_v4 = 0, cidx = 0, var = 0; int member_path = 0;
    size_t first_fn = 0;
    for (size_t i = 5; i < nw;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16; if (!len) return NULL;
        if (op == OpEntryPoint && !entry) entry = w[i + 2];
        else if (op == OpMemberDecorate && len >= 5 && w[i + 3] == 11 && w[i + 4] == 0) { pos_struct = w[i + 1]; pos_member = w[i + 2]; member_path = 1; }
        else if (op == OpDecorate && len >= 4 && w[i + 2] == 11 && w[i + 3] == 0) pos_var = w[i + 1];
        else if (op == OpTypeFloat && w[i + 2] == 32) f32 = w[i + 1];
        else if (op == OpTypeInt && w[i + 2] == 32 && !u32) u32 = w[i + 1];
        else if (op == OpFunction && !first_fn) first_fn = i;
        i += len;
    }
    for (size_t i = 5; i < first_fn;) {   // the vector, pointer, index constant and variable, now the type IDs are known
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (op == OpTypeVector && w[i + 2] == f32 && w[i + 3] == 4) v4 = w[i + 1];
        i += len;
    }
    for (size_t i = 5; i < first_fn;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (op == OpTypePointer && w[i + 2] == Output && w[i + 3] == v4) ptr_v4 = w[i + 1];
        if (op == OpConstant && w[i + 1] == u32 && len == 4 && w[i + 3] == pos_member) cidx = w[i + 2];
        if (op == OpVariable && w[i + 3] == Output) {
            if (!member_path && w[i + 2] == pos_var) var = w[i + 2];
            if (member_path) for (size_t j = 5; j < first_fn;) { uint32_t o2 = w[j] & 0xffff, l2 = w[j] >> 16; if (o2 == OpTypePointer && w[j + 1] == w[i + 1] && w[j + 3] == pos_struct) var = w[i + 2]; j += l2; }
        }
        i += len;
    }
    if (!entry || !f32 || !v4 || !var || (member_path && !u32)) return NULL;
    uint32_t decl[16]; size_t nd = 0;
    if (!ptr_v4) { ptr_v4 = bound++; decl[nd++] = 4u << 16 | OpTypePointer; decl[nd++] = ptr_v4; decl[nd++] = Output; decl[nd++] = v4; }
    if (member_path && !cidx) { cidx = bound++; decl[nd++] = 4u << 16 | OpConstant; decl[nd++] = u32; decl[nd++] = cidx; decl[nd++] = pos_member; }
    uint32_t half = bound++; float hf = 0.5f; uint32_t hb; memcpy(&hb, &hf, 4);
    decl[nd++] = 4u << 16 | OpConstant; decl[nd++] = f32; decl[nd++] = half; decl[nd++] = hb;
    size_t nret = 0; for (size_t i = first_fn; i < nw; i += w[i] >> 16) if ((w[i] & 0xffff) == OpReturn) nret++;
    uint32_t *o = malloc((nw + nd + nret * 40) * 4); size_t n = 0; int in_entry = 0;
    for (size_t i = 0; i < 5; i++) o[n++] = w[i];
    for (size_t i = 5; i < nw;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (i == first_fn) { memcpy(o + n, decl, nd * 4); n += nd; }
        if (op == OpFunction) in_entry = w[i + 2] == entry;
        if (op == OpReturn && in_entry) {
            uint32_t p = var, ld = bound++, z = bound++, ww = bound++, sum = bound++, hz = bound++, nv = bound++;
            if (member_path) { p = bound++; o[n++] = 5u << 16 | OpAccessChain; o[n++] = ptr_v4; o[n++] = p; o[n++] = var; o[n++] = cidx; }
            o[n++] = 4u << 16 | OpLoad; o[n++] = v4; o[n++] = ld; o[n++] = p;
            o[n++] = 5u << 16 | OpCompositeExtract; o[n++] = f32; o[n++] = z; o[n++] = ld; o[n++] = 2;
            o[n++] = 5u << 16 | OpCompositeExtract; o[n++] = f32; o[n++] = ww; o[n++] = ld; o[n++] = 3;
            o[n++] = 5u << 16 | OpFAdd; o[n++] = f32; o[n++] = sum; o[n++] = z; o[n++] = ww;
            o[n++] = 5u << 16 | OpFMul; o[n++] = f32; o[n++] = hz; o[n++] = sum; o[n++] = half;
            o[n++] = 6u << 16 | OpCompositeInsert; o[n++] = v4; o[n++] = nv; o[n++] = hz; o[n++] = ld; o[n++] = 2;
            o[n++] = 3u << 16 | OpStore; o[n++] = p; o[n++] = nv;
        }
        memcpy(o + n, w + i, len * 4); n += len; i += len;
    }
    o[3] = bound; *out_n = n; return o;
}
// Whether a module writes a storage buffer: an OpStore or atomic whose pointer derives (access chains) from a StorageBuffer variable, or a Uniform one whose struct is a BufferBlock.
// Unreal binds read-only storage buffers to many vertex and pixel shaders (FINAL FANTASY RESONANCE: none of the eight that warned writes them)
static int spv_ssbo_writes(const uint32_t *w, size_t nw) {
    enum { OpTypePointer = 32, OpVariable = 59, OpStore = 62, OpAccessChain = 65, OpInBoundsAccessChain = 66, OpPtrAccessChain = 67, OpDecorate = 71, BufferBlock = 3 };
    uint32_t bound = nw > 3 ? w[3] : 0; if (!bound || bound > (1u << 22)) return 1;
    uint8_t *k = calloc(bound, 1); if (!k) return 1;   // bit 1 BufferBlock type, 2 pointer to a storage buffer, 4 derived from a storage buffer variable
    uint32_t *pt = calloc(bound, 4); int wr = 0;
    for (size_t i = 5; i < nw;) { uint32_t op = w[i] & 0xffff, n = w[i] >> 16; if (!n) break;
        if (op == OpDecorate && n >= 3 && w[i + 2] == BufferBlock && w[i + 1] < bound) k[w[i + 1]] |= 1;
        i += n; }
    for (size_t i = 5; i < nw && !wr;) { uint32_t op = w[i] & 0xffff, n = w[i] >> 16; if (!n) break;
        if (op == OpTypePointer && n >= 4 && w[i + 1] < bound && w[i + 3] < bound && (w[i + 2] == 12 || (w[i + 2] == 2 && (k[w[i + 3]] & 1)))) k[w[i + 1]] |= 2;
        else if (op == OpVariable && n >= 4 && w[i + 1] < bound && w[i + 2] < bound && (k[w[i + 1]] & 2)) k[w[i + 2]] |= 4;
        else if ((op == OpAccessChain || op == OpInBoundsAccessChain || op == OpPtrAccessChain) && n >= 4 && w[i + 2] < bound && w[i + 3] < bound && (k[w[i + 3]] & 4)) k[w[i + 2]] |= 4;
        else if (op == OpStore && n >= 3 && w[i + 1] < bound && (k[w[i + 1]] & 4)) wr = 1;
        else if (op >= 227 && op <= 242 && n >= 4 && w[i + 3] < bound && (k[w[i + 3]] & 4)) wr = 1;   // OpAtomic*: result type, result, pointer
        i += n; }
    free(k); free(pt); return wr;
}
// Shaders are loaded on the replay thread and on the pipeline pre-creation thread (pso_precreate): one lock for the table and the loads. Their time is the long-frame lines'
// "shader libraries" (mtl_create_ns): a first-time load reads <key>.spv and <key>.vk.txt from the shader folder (150,000 files in one directory on Android's shared storage)
static pthread_mutex_t g_shmu = PTHREAD_MUTEX_INITIALIZER; static _Atomic uint64_t g_sh_ns;
// A module's load-time rewrites, the same for a vertex shader's padded variant (vs_pad_mod) as for the plain one, so their layouts and behaviour match
static uint32_t *vs_transform(shader_t *s, uint32_t *spv, size_t *nsp, int at_load) {
    size_t ns = *nsp;
    if (s->stage == 0 && vid_fix_on() && (at_load || s->vid)) { size_t nn; uint32_t *w2 = spv_vid_bias(spv, ns / 4, &nn); if (w2) { free(spv); spv = w2; ns = nn * 4; if (at_load) s->vid = 1; } }   // VertexIndex as the game gave it (see spv_vid_bias)
    if (s->stage == 0) { size_t nn; uint32_t *w2 = spv_invariant(spv, ns / 4, &nn); if (w2) { free(spv); spv = w2; ns = nn * 4; } }
    if (s->stage == 0) { size_t nn; uint32_t *w2 = spv_point_size(spv, ns / 4, &nn); if (w2) { free(spv); spv = w2; ns = nn * 4; } }
    if (s->stage == 0 && !g_depth_zero) {   // NVN's GL depth convention (see spv_gl_depth)
        size_t nn; uint32_t *w2 = spv_gl_depth(spv, ns / 4, &nn);
        if (w2) { free(spv); spv = w2; ns = nn * 4; } else if (at_load) LOG("[RED] vk: vertex shader %s: no position written, left as it is for the GL depth convention", s->key);
    }
    spv_set(spv, ns / 4, s->stage == 1 ? 1 : s->stage == 2 ? 2 : 0);   // fragment resources in set 1, geometry in set 2
    *nsp = ns; return spv;
}
// ---------- Pool set (vk_pool) ----------
// A shader whose texture handle the translator traced to no slot indexes the game's texture pool by the handle's texture ID and the sampler pool by its sampler ID (dmsh_build.py
// pool_arrays; the Metal host serves it from pool_gpu). Here those arrays live in one descriptor set shared by every such shader, set SPV_POOL_SET of its layout: a binding per
// texture class (2D, 2D array, 3D, cube; the shader declares the class, spv_pool.h) of pool_cap() entries, entry k = the view of host texture ID tex_base + k when its class is
// the binding's (else that class's black stand-in), and the samplers. One set for the game (the GPU allows 7 bound sets; Tomodachi Life's ground and 188 more of its pixel
// shaders read pools), kept up to date entry by entry (pool_set): the IDs registered since the last sync, the textures released or CPU-written while it held them, the samplers.
// Entries are written while earlier submissions may be pending (UPDATE_AFTER_BIND, UPDATE_UNUSED_WHILE_PENDING; PARTIALLY_BOUND): as on the Switch, a draw reads what the pool
// holds when it runs. A texture the set holds is never evicted and not freed until the set let go of it and the GPU passed that submission (vtex inpool / pool_out)
#include "spv_pool.h"
static VkDescriptorSetLayout g_pool_dsl_; static int g_pool_dsl_tried;
static int g_pool_shaders, g_pool_refused;
static VkDescriptorSetLayout pool_dsl(void) {   // made at the first pool shader (shader_get_, under g_shmu); VK_NULL_HANDLE when the GPU cannot hold it
    if (g_pool_dsl_ || g_pool_dsl_tried) return g_pool_dsl_;
    g_pool_dsl_tried = 1;
    VkPhysicalDeviceVulkan12Properties p12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES}; VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &p12};
    vkGetPhysicalDeviceProperties2(g_pd, &p2);
    uint32_t cap = pool_cap(), need = SPV_PC_N * cap;
    LOG("vk: pool set: %u texture IDs in each of %d classes, %d samplers; the GPU allows (update after bind) sampled images %u per stage / %u per set, samplers %u / %u, %u in all pools; %u bound sets",
        cap, SPV_PC_N, SPV_POOL_SAMP_CAP, p12.maxPerStageDescriptorUpdateAfterBindSampledImages, p12.maxDescriptorSetUpdateAfterBindSampledImages, p12.maxPerStageDescriptorUpdateAfterBindSamplers,
        p12.maxDescriptorSetUpdateAfterBindSamplers, p12.maxUpdateAfterBindDescriptorsInAllPools, g_props.limits.maxBoundDescriptorSets);
    if (p12.maxPerStageDescriptorUpdateAfterBindSampledImages < need || p12.maxDescriptorSetUpdateAfterBindSampledImages < need || p12.maxPerStageDescriptorUpdateAfterBindSamplers < SPV_POOL_SAMP_CAP ||
        p12.maxDescriptorSetUpdateAfterBindSamplers < SPV_POOL_SAMP_CAP || p12.maxUpdateAfterBindDescriptorsInAllPools < need + SPV_POOL_SAMP_CAP || g_props.limits.maxBoundDescriptorSets < SPV_POOL_SET + 1) {
        LOG("[RED] vk: pool set: over the GPU's limits: shaders indexing the pools stay as before (their pools unbound)"); return VK_NULL_HANDLE; }
    VkDescriptorSetLayoutBinding lb[SPV_PC_N + 1]; VkDescriptorBindingFlags bf[SPV_PC_N + 1];
    for (int c = 0; c <= SPV_PC_N; c++) {
        lb[c] = (VkDescriptorSetLayoutBinding){(uint32_t)c, c == SPV_POOL_SAMP ? VK_DESCRIPTOR_TYPE_SAMPLER : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, c == SPV_POOL_SAMP ? SPV_POOL_SAMP_CAP : cap, VK_SHADER_STAGE_FRAGMENT_BIT, NULL};
        bf[c] = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT;
    }
    VkDescriptorSetLayoutBindingFlagsCreateInfo fi = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO, NULL, SPV_PC_N + 1, bf};
    VkDescriptorSetLayoutCreateInfo li = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, &fi, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT, SPV_PC_N + 1, lb};
    VkResult r = vkCreateDescriptorSetLayout(g_dev, &li, NULL, &g_pool_dsl_);
    if (r != VK_SUCCESS) { LOG("[RED] vk: pool set layout failed (%d): shaders indexing the pools stay as before", (int)r); g_pool_dsl_ = VK_NULL_HANDLE; }
    return g_pool_dsl_;
}
// At load (shader_get_): a shader's pool bindings move into the pool set (spv_pool_patch) and out of its own set; anything in the way leaves them as an unknown kind, as before
static uint32_t *pool_shader(shader_t *s, uint32_t *spv, size_t *ns) {
    spv_pool_var pv[16]; int np = 0;
    for (int i = 0; i < s->nb; i++) if (s->b[i].kind == K_TPOOL || s->b[i].kind == K_SPOOL) { if (np < 16) { pv[np].binding = s->b[i].binding; pv[np].samp = s->b[i].kind == K_SPOOL; } np++; }
    if (!np || !spv) return spv;
    const char *why = NULL; uint32_t *o = NULL; size_t nw = *ns / 4;
    if (!g_pool_dev) why = "the GPU lacks partially bound / update-after-bind sampled images or their non-uniform indexing";
    else if (s->stage != 1) why = "not a pixel shader (the pool set is the pixel stage's)";
    else if (np > 16) why = "more than 16 pools";
    else if (!pool_dsl()) why = "no pool set (over the GPU's limits)";
    else if (!(o = spv_pool_patch(spv, &nw, pv, np, pool_cap()))) why = spv_pool_err;
    if (why) {
        for (int i = 0; i < s->nb; i++) if (s->b[i].kind == K_TPOOL || s->b[i].kind == K_SPOOL) s->b[i].kind = K_UNK;
        if (g_pool_refused++ < 16) LOG("[RED] vk: shader %s: its texture / sampler pools stay unbound (%s)", s->key, why);
        return spv;
    }
    int j = 0; for (int i = 0; i < s->nb; i++) if (s->b[i].kind != K_TPOOL && s->b[i].kind != K_SPOOL) s->b[j++] = s->b[i];
    s->nb = j; s->pool = 1;
    if (g_pool_shaders++ < 8) {
        static const char *const cn[] = {"2D", "2D array", "3D", "cube"}; char b[128]; int bo = 0;
        for (int i = 0; i < np && bo < 100; i++) bo += snprintf(b + bo, sizeof b - bo, "%s%s", i ? ", " : "", pv[i].samp ? "samplers" : cn[pv[i].cls]);
        LOG("vk: shader %s: its pools (%s) read the pool set (vk_pool; such shaders read zero before)", s->key, b);
    }
    free(spv); *ns = nw * 4; return o;
}
static shader_t *shader_get_(const uint8_t md[8]);
static shader_t *shader_get(const uint8_t md[8]) {
    pthread_mutex_lock(&g_shmu);
    for (int i = 0; i < g_nsh; i++) if (!memcmp(g_sh[i]->md, md, 8)) { shader_t *s = g_sh[i]; pthread_mutex_unlock(&g_shmu); return s; }
    uint64_t t0 = stat_ns(); shader_t *s = shader_get_(md); memcpy(s->md, md, 8); atomic_fetch_add(&g_sh_ns, stat_ns() - t0);
    pthread_mutex_unlock(&g_shmu); return s;
}
static shader_t *shader_get_(const uint8_t md[8]) {
    char key[17]; for (int i = 0; i < 8; i++) sprintf(key + 2 * i, "%02x", md[i]);
    for (int i = 0; i < g_nsh; i++) if (!strcmp(g_sh[i]->key, key)) return g_sh[i];
    if (!g_vkdir && !(g_vkdir = getenv("DM_VK_DIR"))) die("vk: DM_VK_DIR is not set (the folder of <key>.spv / <key>.vk.txt from dm/android/tools/spv_android.py)");
    char path[1024]; size_t ns, nt;
    snprintf(path, sizeof path, "%s/%s.spv", g_vkdir, key); uint32_t *spv = read_file(path, &ns);
    snprintf(path, sizeof path, "%s/%s.vk.txt", g_vkdir, key); char *txt = (char *)read_file(path, &nt);
    shader_t *s = calloc(1, sizeof *s); strcpy(s->key, key); s->stage = -1;
    g_sh = realloc(g_sh, (g_nsh + 1) * sizeof *g_sh); g_sh[g_nsh++] = s;
    if (!spv || !txt) { LOG("[RED] vk: shader %s not in %s", key, g_vkdir); free(spv); free(txt); return s; }
    for (char *ln = strtok(txt, "\n"); ln; ln = strtok(NULL, "\n")) {
        char a[16], ty[8], kind[16], sk[16]; int b, cnt, reg, off, sreg, soff = -1, l, t;
        if (sscanf(ln, "stage %15s", a) == 1) s->stage = !strcmp(a, "vs") ? 0 : !strcmp(a, "ps") ? 1 : !strcmp(a, "gs") ? 2 : !strcmp(a, "cs") ? 5 : -1;
        else if (sscanf(ln, "tg %u %u %u", &s->tg[0], &s->tg[1], &s->tg[2]) == 3) {}
        else if (!strncmp(ln, "c1 ", 3)) { size_t n = strlen(ln + 3) / 2; s->c1 = malloc(n + 16); s->c1_len = (uint32_t)n; for (size_t i = 0; i < n; i++) sscanf(ln + 3 + 2 * i, "%2hhx", &s->c1[i]); }
        else if (sscanf(ln, "vin %d %d", &l, &t) == 2 && l >= 0 && l < 32) { s->vin[l] = (uint8_t)t; s->vin_mask |= 1u << l; }
        else if (sscanf(ln, "b %d %7s %d %15s %d %d %15s %d %d", &b, ty, &cnt, kind, &reg, &off, sk, &sreg, &soff) >= 8 && s->nb < 64) {   // the 9th field, the sampler's byte offset, is newer
            sbind *e = &s->b[s->nb++]; e->binding = (uint32_t)b; e->count = cnt > 0 ? (uint32_t)cnt : 1; e->reg = (int16_t)reg; e->off = (int32_t)off; e->sreg = (int16_t)sreg; e->soff = (int32_t)soff;
            e->type = !strcmp(ty, "ubo") ? BT_UBO : !strcmp(ty, "ssbo") ? BT_SSBO : !strcmp(ty, "cis") ? BT_CIS : !strcmp(ty, "tbuf") ? BT_TBUF : !strcmp(ty, "utb") ? BT_UTB :
                      !strcmp(ty, "stb") ? BT_STB : !strcmp(ty, "simg") ? BT_SIMG : !strcmp(ty, "tex") ? BT_TEX : BT_SAMP;
            e->kind = !strcmp(kind, "cbuf") ? K_CBUF : !strcmp(kind, "c1") ? K_C1 : !strcmp(kind, "support") ? K_SUPPORT : !strcmp(kind, "drvcb") ? K_DRVCB : !strcmp(kind, "tex") ? K_TEX :
                      !strcmp(kind, "img") ? K_IMG : !strcmp(kind, "sbuf") ? K_SBUF : !strcmp(kind, "tbuf") ? K_TBUF : !strcmp(kind, "ubo_ptr") ? K_UBOP : !strcmp(kind, "tex_ubo") ? K_TEXU :
                      pool_on() && !strcmp(kind, "tex_pool") ? K_TPOOL : pool_on() && !strcmp(kind, "samp_pool") ? K_SPOOL : K_UNK;
            e->skind = !strcmp(sk, "samp") ? S_SAMP : !strcmp(sk, "samp_tex") ? S_SAMP_TEX : !strcmp(sk, "samp_id") ? S_SAMP_ID : !strcmp(sk, "samp_ubo") ? S_SAMP_UBO : S_NONE;
#ifndef DM_GAME_TEX_DYN_SLOTS
#define DM_GAME_TEX_DYN_SLOTS 0
#endif
            int dynarr = DM_GAME_TEX_DYN_SLOTS && e->count > 1 && e->reg < 0 && (e->type == BT_UTB || e->type == BT_TBUF || e->type == BT_SAMP || e->type == BT_TEX || e->type == BT_CIS);
            if (dynarr) s->dyn = 1;
            else if (e->kind == K_UNK) LOG("[RED] vk: shader %s binding %d has kind %s, not implemented (reads zero)", key, b, kind);
            if (e->kind == K_TEXU && e->count > 1) LOG("vk: shader %s binding %d: an array of %u handles in a uniform buffer (texu_elem: element k at byte %d + 8k of slot %d)", key, b, e->count, e->off, e->reg);
        }
    }
    free(txt);
    if (pool_on()) spv = pool_shader(s, spv, &ns);   // tex_pool / samp_pool: into the pool set, or back to the unknown kind
    s->ssbo_wr = spv_ssbo_writes(spv, ns / 4);
    for (size_t i = 5; i + 4 < ns / 4;) { uint32_t op = spv[i] & 0xffff, n = spv[i] >> 16; if (!n) break;   // BuiltIn Layer (9): OpDecorate / OpMemberDecorate
        if ((op == 71 && n >= 4 && spv[i + 2] == 11 && spv[i + 3] == 9) || (op == 72 && n >= 5 && spv[i + 3] == 11 && spv[i + 4] == 9)) s->layer_out = 1;
        i += n; }
    s->io_mask = s->stage == 0 ? spv_io(spv, ns / 4, 3, NULL, &s->io_complex) : s->stage == 1 ? spv_io(spv, ns / 4, 1, s->in_type, &s->io_complex) : 0;   // outputs / inputs (vs_pad_mod)
    spv = vs_transform(s, spv, &ns, 1);
    VkShaderModuleCreateInfo mi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, ns, spv};
    VkResult r = vkCreateShaderModule(g_dev, &mi, NULL, &s->mod); free(spv);
    if (r != VK_SUCCESS) { LOG("[RED] vk: shader %s rejected (%d)", key, (int)r); s->mod = VK_NULL_HANDLE; }
    VkDescriptorSetLayoutBinding lb[64]; VkShaderStageFlags sf = s->stage == 0 ? VK_SHADER_STAGE_VERTEX_BIT : s->stage == 1 ? VK_SHADER_STAGE_FRAGMENT_BIT : s->stage == 2 ? VK_SHADER_STAGE_GEOMETRY_BIT : VK_SHADER_STAGE_COMPUTE_BIT;
    static const VkDescriptorType DT[] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
                                          VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_SAMPLER};
    // Uniform buffers the ring holds (the game's, c1, driver segments) are dynamic, as many as a third of the GPU's limit per pipeline layout allows (three sets), the game's first;
    // only when the bindings are in ascending order (dynamic offsets are given in binding order)
    { uint32_t lim = g_props.limits.maxDescriptorSetUniformBuffersDynamic / 3, nd = 0; int asc = 1;
      for (int i = 1; i < s->nb; i++) if (s->b[i].binding <= s->b[i - 1].binding) asc = 0;
      for (int pass = 0; pass < 2 && asc; pass++) for (int i = 0; i < s->nb && nd < lim; i++) {
          const sbind *b = &s->b[i]; if (b->type != BT_UBO || b->count != 1 || s->dynu[i]) continue;
          if (pass == 0 ? b->kind == K_CBUF : (b->kind == K_C1 || b->kind == K_DRVCB)) { s->dynu[i] = 1; nd++; } } }
    for (int i = 0; i < s->nb; i++) lb[i] = (VkDescriptorSetLayoutBinding){s->b[i].binding, s->dynu[i] ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC : DT[s->b[i].type], s->b[i].count, sf, NULL};
    VkDescriptorSetLayoutCreateInfo li = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, (uint32_t)s->nb, lb};
    VKCHECK(vkCreateDescriptorSetLayout(g_dev, &li, NULL, &s->dsl));
    return s;
}
static VkDescriptorSetLayout g_empty_dsl;
typedef struct { shader_t *vs, *fs, *gs; VkPipelineLayout pl; } prog_vk;
static prog_vk *g_pl; static int g_npl;
static pthread_mutex_t g_plmu = PTHREAD_MUTEX_INITIALIZER;
static VkPipelineLayout layout_get_(shader_t *vs, shader_t *fs, shader_t *gs);
static VkPipelineLayout layout_get(shader_t *vs, shader_t *fs, shader_t *gs) {   // the last answer of this thread first (a layout is never destroyed): the list was scanned for every draw
    static __thread shader_t *lv, *lf, *lg; static __thread VkPipelineLayout ll;
    if (ll && lv == vs && lf == fs && lg == gs) return ll;
    pthread_mutex_lock(&g_plmu); VkPipelineLayout l = layout_get_(vs, fs, gs); pthread_mutex_unlock(&g_plmu);
    if (l) { lv = vs; lf = fs; lg = gs; ll = l; }
    return l;
}
static VkPipelineLayout layout_get_(shader_t *vs, shader_t *fs, shader_t *gs) {
    for (int i = 0; i < g_npl; i++) if (g_pl[i].vs == vs && g_pl[i].fs == fs && g_pl[i].gs == gs) return g_pl[i].pl;
    if (!g_empty_dsl) { VkDescriptorSetLayoutCreateInfo li = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; VKCHECK(vkCreateDescriptorSetLayout(g_dev, &li, NULL, &g_empty_dsl)); }
    VkDescriptorSetLayout sl[4] = {vs ? vs->dsl : g_empty_dsl, fs ? fs->dsl : g_empty_dsl, gs ? gs->dsl : g_empty_dsl, fs && fs->pool ? pool_dsl() : VK_NULL_HANDLE};   // set 3: the pool set
    VkPushConstantRange pr = {VK_SHADER_STAGE_VERTEX_BIT, 0, 4};   // a vertex shader reading VertexIndex: the shift draw() made (spv_vid_bias)
    VkPipelineLayoutCreateInfo pi = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, fs && fs->pool ? 4 : gs ? 3 : fs ? 2 : 1, sl, vs && vs->vid ? 1 : 0, vs && vs->vid ? &pr : NULL};
    VkPipelineLayout pl; VKCHECK(vkCreatePipelineLayout(g_dev, &pi, NULL, &pl));
    g_pl = realloc(g_pl, (g_npl + 1) * sizeof *g_pl); g_pl[g_npl++] = (prog_vk){vs, fs, gs, pl};
    return pl;
}

// ---------- Render passes: dynamic rendering, or a render pass + framebuffer made for each pass ----------
typedef struct { int n; VkImageView cv[8]; VkFormat cf[8]; VkAttachmentLoadOp cl[8]; VkClearColorValue cc[8];
                 VkImageView dv; VkFormat df; int stencil; VkAttachmentLoadOp dl, sl; VkClearDepthStencilValue dc; uint32_t w, h; VkImageLayout layout; uint32_t layers; } rp_desc;
static struct { uint64_t key; VkRenderPass rp; } g_rpc[512]; static int g_nrpc; static pthread_mutex_t g_rpmu = PTHREAD_MUTEX_INITIALIZER;
// The render pass of these attachments (cached; also the "compatible" one pipelines are made against: load actions do not affect compatibility)
static VkRenderPass rp_get(const VkFormat *cf, const VkAttachmentLoadOp *cl, int n, VkFormat df, int stencil, VkAttachmentLoadOp dl, VkAttachmentLoadOp sl, VkImageLayout layout) {
    uint64_t key = 0xcbf29ce484222325ull;
    #define MIX(v) key = (key ^ (uint64_t)(v)) * 0x100000001b3ull
    for (int i = 0; i < n; i++) { MIX(cf[i]); MIX(cl[i]); } MIX(n); MIX(df); MIX(stencil); MIX(dl); MIX(sl); MIX(layout);
    #undef MIX
    pthread_mutex_lock(&g_rpmu);
    for (int i = 0; i < g_nrpc; i++) if (g_rpc[i].key == key) { VkRenderPass r = g_rpc[i].rp; pthread_mutex_unlock(&g_rpmu); return r; }
    VkAttachmentDescription at[9]; VkAttachmentReference cr[8], dr = {VK_ATTACHMENT_UNUSED, layout}; uint32_t na = 0;
    for (int i = 0; i < n; i++) {
        if (!cf[i]) { cr[i] = (VkAttachmentReference){VK_ATTACHMENT_UNUSED, layout}; continue; }
        at[na] = (VkAttachmentDescription){0, cf[i], VK_SAMPLE_COUNT_1_BIT, cl[i], VK_ATTACHMENT_STORE_OP_STORE, VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE, layout, layout};
        cr[i] = (VkAttachmentReference){na++, layout};
    }
    if (df) { at[na] = (VkAttachmentDescription){0, df, VK_SAMPLE_COUNT_1_BIT, dl, VK_ATTACHMENT_STORE_OP_STORE, stencil ? sl : VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                                stencil ? VK_ATTACHMENT_STORE_OP_STORE : VK_ATTACHMENT_STORE_OP_DONT_CARE, layout, layout}; dr.attachment = na++; }
    VkSubpassDescription sp = {0, VK_PIPELINE_BIND_POINT_GRAPHICS, 0, NULL, (uint32_t)n, cr, NULL, df ? &dr : NULL, 0, NULL};
    VkRenderPassCreateInfo ci = {VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO, NULL, 0, na, at, 1, &sp, 0, NULL};
    VkRenderPass r; VKCHECK(vkCreateRenderPass(g_dev, &ci, NULL, &r));
    if (g_nrpc < 512) g_rpc[g_nrpc++] = (typeof(g_rpc[0])){key, r};
    pthread_mutex_unlock(&g_rpmu);
    return r;
}
// Begins rendering into d's attachments; without dynamic rendering the framebuffer it makes is returned, for the caller to destroy once the GPU is done with it
static VkFramebuffer rp_begin(VkCommandBuffer cb, const rp_desc *d) {
    if (g_dynrend) {
        VkRenderingAttachmentInfo ca[8], da = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO}, sa = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        for (int i = 0; i < d->n; i++) ca[i] = (VkRenderingAttachmentInfo){VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, NULL, d->cv[i], d->layout, 0, NULL, 0,
                                                                          d->cv[i] ? d->cl[i] : VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_STORE, {.color = d->cc[i]}};
        if (d->dv) { da.imageView = d->dv; da.imageLayout = d->layout; da.loadOp = d->dl; da.storeOp = VK_ATTACHMENT_STORE_OP_STORE; da.clearValue.depthStencil = d->dc;
                     if (d->stencil) { sa = da; sa.loadOp = d->sl; } }
        VkRenderingInfo ri = {VK_STRUCTURE_TYPE_RENDERING_INFO, NULL, 0, {{0, 0}, {d->w, d->h}}, d->layers ? d->layers : 1, 0, (uint32_t)d->n, ca, d->dv ? &da : NULL, d->dv && d->stencil ? &sa : NULL};
        p_begin_rendering(cb, &ri);
        return VK_NULL_HANDLE;
    }
    VkFormat cf[8]; VkImageView views[9]; VkClearValue cv[9]; uint32_t nv = 0;
    for (int i = 0; i < d->n; i++) { cf[i] = d->cv[i] ? d->cf[i] : VK_FORMAT_UNDEFINED; if (d->cv[i]) { cv[nv].color = d->cc[i]; views[nv++] = d->cv[i]; } }
    if (d->dv) { cv[nv].depthStencil = d->dc; views[nv++] = d->dv; }
    VkRenderPass rp = rp_get(cf, d->cl, d->n, d->dv ? d->df : VK_FORMAT_UNDEFINED, d->stencil, d->dl, d->sl, d->layout);
    VkFramebufferCreateInfo fi = {VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, NULL, 0, rp, nv, views, d->w, d->h, d->layers ? d->layers : 1};
    VkFramebuffer fb; VKCHECK(vkCreateFramebuffer(g_dev, &fi, NULL, &fb));
    VkRenderPassBeginInfo bi = {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, NULL, rp, fb, {{0, 0}, {d->w, d->h}}, nv, cv};
    vkCmdBeginRenderPass(cb, &bi, VK_SUBPASS_CONTENTS_INLINE);
    return fb;
}
static void occ_close(VkCommandBuffer cb);
static void rp_end(VkCommandBuffer cb) { occ_close(cb); if (g_dynrend) p_end_rendering(cb); else vkCmdEndRenderPass(cb); }

// ---------- Replay state ----------
#define NTEX 256
static struct {
    const prog_t *bst[5], *cprog;   // the program bound to each graphics stage (NVN stage bits 1 vertex, 2 fragment, 4 geometry, 8 / 16 tessellation): games bind a program to some stages and NULL to others
    blend_st blend[8]; color_st color; chmask_st chmask; ms_st ms; poly_st poly; ds_st ds;
    va_st attr[16]; int nattr; vs_st stream[16]; int nstream; NVNbufferRange vbuf[16];
    p_bind_buf ubo[6][16], ssbo[6][16];
    uint64_t tex[6][NTEX], samp[6][128], img[6][32];
    uint32_t st_vmask[2], st_wmask[2], st_ref[2];
    p_poly_offset po; float vp[16][4]; float dr[2]; int has_vp; p_scissor sc; int has_sc;
    float blend_color[4]; int64_t copy_stride, copy_img_stride;
    int tex_base, samp_base;   // host-table bases of the texture / sampler pool bound (OP_SET_TEX_POOL / OP_SET_SAMP_POOL), for handles read from uniform buffers (K_TEXU)
    float line_w; p_prim_restart restart;   // SetLineWidth, SetPrimitiveRestart
} S;
static int g_depth_zero = 1, g_origin_ll;
static struct {   // the current render targets and the open dynamic rendering; clears that come before the pass's first draw become its load actions (a tiler then never reads the old contents)
    int n; vtex *ct[8]; VkImageView cv[8]; VkFormat cf[8]; vtex *dt; VkImageView dv; uint32_t w, h; int open;   // cf: each colour attachment's view format (rt_view_fmt)
    int merged;   // the open pass went on across a SetRenderTargets (rt_hold_scan, or the same targets again): a draw sampling one of its attachments ends it first (draw_reads_rt)
    uint8_t pw_c[8], pw_d; rp_desc rpd;   // DM_VK_STORE_NONE: what this open pass wrote (draws, clears, its CLEAR loads), and its description (begun at its end)
    uint32_t clv[8], cly[8]; uint8_t cwr[8], cdrawn[8];   // each colour attachment's level and layer, and whether a draw wrote it since its last same-pass snapshot (fb_prepare)
    uint8_t ddrawn;   // the depth / stencil attachment was cleared or written in this pass (draw_reads_rt: an attachment only loaded reads the same from memory)
    uint8_t clr_c[8], clr_d, clr_s; VkClearColorValue cc[8]; float cd; uint32_t cs;
    // Layered rendering: a draw whose vertex (or geometry) shader writes the layer runs in a pass over every layer of the attachments (3D: their depth slices) from the attached one on,
    // through 2D array views (tex_view rt 3). Unreal renders its 32x32x32 colour grading LUT this way, one instance per slice; with a one-slice view all instances landed in slice 0
    // and the tonemapper's lookups read black (FINAL FANTASY RESONANCE's fields: black with white highlights; Pokémon Legends: Z-A on the Metal host, nvn_mtl.m)
    const p_set_rt *rt; int layered, want_layered, lok; VkImageView lcv[8], ldv; uint32_t nlayers;
} P;
static int pass_begin(void);
// A linear texture the GPU rendered into is read by the CPU from its storage: Super Mario Odyssey renders the scene's luminance into a 2x1 linear RG11B10F target and computes
// the exposure from it on the CPU; left in the image, the CPU read stale bytes and the exposure swung between a black and a yellow picture. The Metal host renders such
// textures in place on the pool's buffer; here the pass's result is copied into the ring and back into the storage when the submission completes (post kind 3)
static uint64_t g_lin_wb;
static void lin_writeback(vtex *v) {
    size_t n = (size_t)v->linrow * v->h;
    VkDeviceSize o = ring_alloc(n, 16); if (o == VK_WHOLE_SIZE) return;
    VkBufferImageCopy bc = {o, v->linrow / v->linbpb, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {v->w, v->h, 1}};
    vkCmdCopyImageToBuffer(C->cb, v->img, VK_IMAGE_LAYOUT_GENERAL, C->ring, 1, &bc);
    size_t packed = (size_t)v->w * v->linbpb;   // a pitch past the texels: only the texels go back, the bytes between rows stay the guest's
    if (v->linrow > packed) post_add((post_t){6, (void *)v->mem, v->linrow, v->h, 0, (uint32_t)packed, C->ringp + o, n});
    else post_add((post_t){3, (void *)v->mem, 0, 0, 0, 0, C->ringp + o, n});
    if (!g_lin_wb++) LOG("vk: linear render target %ux%u format %d goes back to its storage after each pass (serial %llu)", v->w, v->h, (int)v->f.f, (unsigned long long)v->serial);
}
// DM_VK_PASS_BARRIER=1: a pass's end barrier leaves the vertex shaders out of its destination, so the next pass's vertex work (the Adreno's binning) may overlap this pass's
// fragments instead of every pass end draining the GPU (59 a frame in the castle). A vertex shader that samples a texture written by such a pass, in the same barrier epoch, gets
// a barrier for it first (draw_vs_hazard: the open pass is ended for it). Submissions still begin with a full barrier. Default 0 (off) until measured
static uint64_t g_nbar_narrow, g_nbar_vs;
// DM_VK_DISPATCH_DEFER=1: a dispatch met while a render pass is open is recorded after that pass instead of ending it, when it reads none of the pass's attachments. FINAL FANTASY
// RESONANCE's castle: a translucency-lighting-volume injection (four 32x32x32 storage images) between consecutive deferred-light draws split the lights into 7 passes a frame,
// each loading and storing the scene colour and depth (~0.7 ms each). The dispatch's resources are resolved when it comes (its descriptor set, its uniform copies, the ranges it
// writes registered as written by this submission); only its recording waits. A draw in the pass that reads what a deferred dispatch writes -- a buffer range it writes (gw_find)
// or a storage image of it -- ends the pass first and the dispatches are recorded before it (DRAWQ). Every pass end records them, in their order. Default 0 (off) until measured
typedef struct { shader_t *cs; VkDescriptorSet ds; uint32_t dyno[64], ndyno, g[3], zero_tb; } dfr_t;
enum { NDEFER = 64 };
static dfr_t g_dfr[NDEFER]; static int g_ndefer; static uint64_t g_defer_gen = 1, g_ndefer_n, g_ndefer_haz;
static uint64_t fnv(const void *p, size_t n);
static int defer_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_DISPATCH_DEFER"); v = e && atoi(e) != 0; if (v) LOG("DM_VK_DISPATCH_DEFER=1: dispatches inside an open pass are recorded after it"); } return v; }
static void defer_flush(void) {   // after the pass ended (pass_end): the deferred dispatches, in order, each followed by its barrier
    if (!g_ndefer) return;
    for (int i = 0; i < g_ndefer; i++) { dfr_t *d = &g_dfr[i];
        vkCmdBindPipeline(C->cb, VK_PIPELINE_BIND_POINT_COMPUTE, d->cs->cpp);
        vkCmdBindDescriptorSets(C->cb, VK_PIPELINE_BIND_POINT_COMPUTE, d->cs->cpl, 0, 1, &d->ds, d->ndyno, d->dyno);
        { char b[64]; snprintf(b, sizeof b, "cs %s", d->cs->key); g_tsig_next = fnv(d->cs->key, strlen(d->cs->key)) | 1; gsig_name(g_tsig_next, b); }
        gpu_mark(1, 0); vkCmdDispatch(C->cb, d->g[0], d->g[1], d->g[2]); gpu_mark(1, 1);
        trace_add('c', d->cs->key, NULL, d->g[0], d->g[1], d->g[2], 0, d->zero_tb);
        barrier_cs(C->cb);
    }
    lk_(&g_gwmu, LW_GWMU); for (int k = 0; k < g_ngw; k++) g_gw[k].dfr = 0; pthread_mutex_unlock(&g_gwmu);
    g_ndefer = 0; g_defer_gen++;
}
static int pass_barrier_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_PASS_BARRIER"); v = e && atoi(e) != 0; if (v) LOG("DM_VK_PASS_BARRIER=1: pass ends do not wait for vertex shaders (their reads of the pass's targets get a barrier)"); } return v; }
static const uint8_t *g_rt_hold;   // see rt_hold_scan
// DM_VK_STORE_NONE=1 (vk19.51, opt-in; dynamic rendering only). Every pass stored every attachment and loaded it unless cleared: on the Adreno's tiles, a store writes the whole
// attachment back from tile memory even when the pass left it alone. The store has to be chosen when rendering begins, before the pass's draws are known, so the pass is recorded
// into a secondary command buffer instead, and at its end the primary begins rendering with STORE for the attachments the pass wrote (a draw with their channels, a clear, a
// CLEAR load) and NONE (the memory left as it is) for the others, executes the secondary and ends. Loads stay as they were. Each secondary starts with no bound state
static int store_none_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_STORE_NONE"); v = e && atoi(e) != 0 && g_dynrend && g_props.apiVersion >= VK_API_VERSION_1_3;
    if (v && defer_on()) { v = 0; LOG("[RED] vk: DM_VK_STORE_NONE is off with DM_VK_DISPATCH_DEFER (a deferred dispatch splits a pass in the middle of a draw's recording)"); }
    if (v) LOG("vk: DM_VK_STORE_NONE=1: passes recorded into secondary command buffers, unwritten attachments not stored"); } return v; }
static uint64_t g_sn_passes, g_sn_none_c, g_sn_all_c, g_sn_none_d, g_sn_all_d;
static void ds_last_newcb(void);   // the state skip's record of what this command buffer has bound: none (a new secondary, or the primary after one ran)
static VkCommandBuffer sec_begin(const rp_desc *d, VkSampleCountFlagBits samples) {
    if (C->nsec == C->csec) { int add = 16; C->sec = realloc(C->sec, (size_t)(C->csec + add) * sizeof *C->sec);
        VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, C->pool, VK_COMMAND_BUFFER_LEVEL_SECONDARY, (uint32_t)add}; VKCHECK(vkAllocateCommandBuffers(g_dev, &ai, C->sec + C->csec)); C->csec += add; }
    VkCommandBuffer sb = C->sec[C->nsec++];
    VkFormat cf[8]; for (int i = 0; i < d->n; i++) cf[i] = d->cv[i] ? d->cf[i] : VK_FORMAT_UNDEFINED;
    VkCommandBufferInheritanceRenderingInfo ir = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_RENDERING_INFO, NULL, 0, 0, (uint32_t)d->n, cf, d->dv ? d->df : VK_FORMAT_UNDEFINED, d->dv && d->stencil ? d->df : VK_FORMAT_UNDEFINED, samples};
    VkCommandBufferInheritanceInfo ih = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO, &ir, VK_NULL_HANDLE, 0, VK_NULL_HANDLE, VK_FALSE, 0, 0};
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT | VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT, &ih};
    VKCHECK(vkBeginCommandBuffer(sb, &bi));
    return sb;
}
static void sec_end(void) {   // the open pass's secondary ends; the primary renders it with the stores it needs
    occ_close(C->cb); VKCHECK(vkEndCommandBuffer(C->cb));
    VkCommandBuffer sb = C->cb; C->cb = C->prim; C->prim = VK_NULL_HANDLE;
    const rp_desc *d = &P.rpd;
    VkRenderingAttachmentInfo ca[8], da = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO}, sa = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    for (int i = 0; i < d->n; i++) {
        int w = P.pw_c[i] || d->cl[i] == VK_ATTACHMENT_LOAD_OP_CLEAR; if (d->cv[i]) { if (w) g_sn_all_c++; else g_sn_none_c++; }
        ca[i] = (VkRenderingAttachmentInfo){VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, NULL, d->cv[i], d->layout, 0, NULL, 0, d->cv[i] ? d->cl[i] : VK_ATTACHMENT_LOAD_OP_DONT_CARE, w ? VK_ATTACHMENT_STORE_OP_STORE : VK_ATTACHMENT_STORE_OP_NONE, {.color = d->cc[i]}};
    }
    if (d->dv) { int w = P.pw_d || d->dl == VK_ATTACHMENT_LOAD_OP_CLEAR || d->sl == VK_ATTACHMENT_LOAD_OP_CLEAR; if (w) g_sn_all_d++; else g_sn_none_d++;
                 da.imageView = d->dv; da.imageLayout = d->layout; da.loadOp = d->dl; da.storeOp = w ? VK_ATTACHMENT_STORE_OP_STORE : VK_ATTACHMENT_STORE_OP_NONE; da.clearValue.depthStencil = d->dc;
                 if (d->stencil) { sa = da; sa.loadOp = d->sl; } }
    VkRenderingInfo ri = {VK_STRUCTURE_TYPE_RENDERING_INFO, NULL, VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT, {{0, 0}, {d->w, d->h}}, d->layers ? d->layers : 1, 0, (uint32_t)d->n, ca, d->dv ? &da : NULL, d->dv && d->stencil ? &sa : NULL};
    p_begin_rendering(C->cb, &ri); vkCmdExecuteCommands(C->cb, 1, &sb); p_end_rendering(C->cb);
    g_sn_passes++; ds_last_newcb();
}
static uint64_t g_npass_clr;   // render passes begun only for clears no draw followed (pass_end_; the GPU line, vk-t8)
static void pass_end_(int line) {
    g_rt_hold = NULL; g_h0_cpass = 0;
    if (P.open) gsig_end(g_cur_sig, line);
    if (!P.open) { int pend = P.clr_d || P.clr_s; for (int i = 0; i < 8; i++) pend |= P.clr_c[i]; if (pend && pass_begin()) g_npass_clr++; }   // clears with no draw after them still happen (a pass of their own: counted, the GPU line)
    if (P.open) {
        if (C->prim) sec_end(); else rp_end(C->cb);
        P.open = 0; gpu_mark(0, 1);
        // what the pass wrote is read by later passes, samples and copies; their vertex work may start before it ends
        VkPipelineStageFlags dst = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                   VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;   // compute too: a dispatch that reads what the pass drew (Unreal's HZB from depth)
        if (pass_barrier_on()) {   // vertex shaders left out: the next pass's vertex work may overlap this one's fragments; the attachments written are marked for draw_vs_hazard
            dst &= ~(VkPipelineStageFlags)VK_PIPELINE_STAGE_VERTEX_SHADER_BIT; g_nbar_narrow++;
            for (int i = 0; i < P.n; i++) if (P.ct[i] && P.cdrawn[i]) { P.ct[i]->vhaz = 1; P.ct[i]->vhaz_epoch = g_vbar_epoch; }
            if (P.dt && P.ddrawn) { P.dt->vhaz = 1; P.dt->vhaz_epoch = g_vbar_epoch; }
        } else g_vbar_epoch++;
        VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT};
        vkCmdPipelineBarrier(C->cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, dst, 0, 1, &mb, 0, NULL, 0, NULL);
        defer_flush();   // the dispatches deferred while it was open (defer_on)
        int wb = 0;
        for (int i = 0; i < P.n; i++) if (P.ct[i] && P.ct[i]->linrow && P.cdrawn[i] && !P.clv[i] && !P.cly[i]) { lin_writeback(P.ct[i]); wb = 1; }
        memset(P.cdrawn, 0, sizeof P.cdrawn); P.ddrawn = 0;
        if (wb) barrier_all(C->cb);
    }
}
static int g_pass_open_(void) { return P.open; }
static uint64_t g_npass, g_nheld, g_nsame;   // render passes begun / SetRenderTargets detours held / same-target SetRenderTargets kept (the GPU line)
static int g_after_split;   // the next pass begins where a held pass was split (draw_reads_rt): its signature says so
static int pass_begin(void) {
    if (P.open) return 1;
    if (!P.n && !P.dt) return 0;
    rp_desc d = {.n = P.n, .w = P.w, .h = P.h, .layout = VK_IMAGE_LAYOUT_GENERAL, .layers = 1};
    if (P.want_layered && !P.lok && P.rt) {   // the attachments' layered views, made at the first layered draw after set_rt
        const p_set_rt *r = P.rt; P.lok = 1; P.nlayers = UINT32_MAX;
        for (int i = 0; i < P.n; i++) if (P.ct[i] && r->color[i]) { vtex *v = P.ct[i]; P.lcv[i] = tex_view(v, r->color[i], &r->cview[i], r->has_cview[i], -1, 3);
            uint32_t have = v->target == NVN_TEXTURE_TARGET_3D ? (v->d >> P.clv[i] ? v->d >> P.clv[i] : 1) - P.cly[i] : v->layers - P.cly[i]; if (have < P.nlayers) P.nlayers = have; }
        if (P.dt && r->depth) { vtex *v = P.dt; uint32_t b0 = r->has_dview && r->dview.set_layers ? r->dview.min_layer : 0; P.ldv = tex_view(v, r->depth, &r->dview, r->has_dview, -1, 3);
            if (v->layers - b0 < P.nlayers) P.nlayers = v->layers - b0; }
        if (P.nlayers == UINT32_MAX || P.nlayers > g_props.limits.maxFramebufferLayers) P.nlayers = P.nlayers == UINT32_MAX ? 1 : g_props.limits.maxFramebufferLayers;
        static int n; if (n++ < 4) LOG("vk: layered pass: %u layers (the vertex shader writes the layer)", P.nlayers);
    }
    P.layered = P.want_layered && P.lok && P.nlayers > 1;
    if (P.layered) d.layers = P.nlayers;
    for (int i = 0; i < P.n; i++) { d.cv[i] = P.layered && P.lcv[i] ? P.lcv[i] : P.cv[i]; d.cf[i] = P.ct[i] ? (P.cf[i] ? P.cf[i] : P.ct[i]->f.f) : VK_FORMAT_UNDEFINED; d.cl[i] = P.clr_c[i] ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD; d.cc[i] = P.cc[i]; if (P.clr_c[i]) P.cdrawn[i] = 1; }
    if (P.dt) { d.dv = P.layered && P.ldv ? P.ldv : P.dv; d.df = P.dt->f.f; d.stencil = P.dt->f.stencil; d.dl = P.clr_d ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
                d.sl = P.clr_s ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD; d.dc = (VkClearDepthStencilValue){P.cd, P.cs}; if (P.clr_d || P.clr_s) P.ddrawn = 1; }
    memset(P.clr_c, 0, sizeof P.clr_c); P.clr_d = P.clr_s = 0;
    if (g_plog) { char b[512]; int o = 0; for (int i = 0; i < P.n; i++) o += snprintf(b + o, sizeof b - o, " c%d=%llu/%d%s", i, P.ct[i] ? (unsigned long long)P.ct[i]->serial : 0ull, (int)d.cf[i], d.cl[i] == VK_ATTACHMENT_LOAD_OP_CLEAR ? "/clr" : "");
                  if (P.dt) snprintf(b + o, sizeof b - o, " d=%llu/%d%s%s", (unsigned long long)P.dt->serial, (int)d.df, d.dl == VK_ATTACHMENT_LOAD_OP_CLEAR ? "/clr" : "", d.sl == VK_ATTACHMENT_LOAD_OP_CLEAR ? "/sclr" : "");
                  LOG("plog: pass %ux%u%s", P.w, P.h, b); }
    {   // this pass's signature for the per-pass GPU times: size, attachments (format, cleared), layered, and whether a held pass's split began it
        uint64_t h = 0xcbf29ce484222325ull;
        #define SMIX(v) h = (h ^ (uint64_t)(v)) * 0x100000001b3ull
        SMIX(P.w); SMIX(P.h); SMIX(P.n); SMIX(d.layers); SMIX(g_after_split);
        for (int i = 0; i < P.n; i++) { SMIX(d.cf[i]); SMIX(d.cl[i]); }
        if (P.dt) { SMIX(d.df); SMIX(d.dl); SMIX(d.sl); }
        #undef SMIX
        g_tsig_next = h & ~1ull;
        static uint64_t seen[256]; uint32_t si = (uint32_t)(g_tsig_next >> 7) % 256;
        if (seen[si] != g_tsig_next) {   // a signature not described lately: its description (the snprintf for every pass was 1% of the render thread)
            seen[si] = g_tsig_next; char b[120]; int o = snprintf(b, sizeof b, "%ux%u", P.w, P.h);
            for (int i = 0; i < P.n && o < (int)sizeof b; i++) o += snprintf(b + o, sizeof b - o, " c%d=%d%s", i, (int)d.cf[i], d.cl[i] == VK_ATTACHMENT_LOAD_OP_CLEAR ? "/clr" : "");
            if (P.dt && o < (int)sizeof b) o += snprintf(b + o, sizeof b - o, " d=%d%s%s", (int)d.df, d.dl == VK_ATTACHMENT_LOAD_OP_CLEAR ? "/clr" : "", d.sl == VK_ATTACHMENT_LOAD_OP_CLEAR ? "/sclr" : "");
            if (d.layers > 1 && o < (int)sizeof b) o += snprintf(b + o, sizeof b - o, " layers %u", d.layers);
            if (g_after_split && o < (int)sizeof b) o += snprintf(b + o, sizeof b - o, " (after a split)");
            gsig_name(g_tsig_next, b);
        }
        g_after_split = 0; g_cur_sig = g_tsig_next;
        if (P.n >= 4) { g_scene_w = P.w; g_scene_h = P.h; }
    }
    gpu_mark(0, 0); g_npass++; P.merged = 0;
    memset(P.pw_c, 0, sizeof P.pw_c); P.pw_d = 0;
    if (store_none_on()) {   // recorded into a secondary, begun at the pass's end (sec_end)
        VkSampleCountFlagBits sc = P.n && P.ct[0] ? P.ct[0]->ci.samples : P.dt ? P.dt->ci.samples : VK_SAMPLE_COUNT_1_BIT; if (!sc) sc = VK_SAMPLE_COUNT_1_BIT;
        P.rpd = d; C->prim = C->cb; C->cb = sec_begin(&d, sc); ds_last_newcb(); P.open = 1; return 1;
    }
    VkFramebuffer fb = rp_begin(C->cb, &d); P.open = 1;
    if (fb) { if (C->nfb == C->cfb) { C->cfb = C->cfb ? 2 * C->cfb : 256; C->fb = realloc(C->fb, C->cfb * sizeof *C->fb); } C->fb[C->nfb++] = fb; }
    return 1;
}
// Render targets held across a detour (as nvn_mtl.m rt_hold_scan, DM_VK_RT_HOLD=0 turns it off): Unreal's parallel base pass begins each of its command lists by setting other
// render targets (another count) and then the six G-buffer targets again, with only state changes in between (FINAL FANTASY RESONANCE's battle: the G-buffer drawn in three passes a
// frame, each storing six 1384x780 targets and depth and loading them again, a full tile load and store on the Adreno). A SetRenderTargets that would end the open pass looks ahead in
// replay order -- called lists, the callers after a list ends, the submission's remaining command handles: when only state changes lie between it and a SetRenderTargets back to the open
// pass's targets, the open pass goes on and the SetRenderTargets in between are not replayed. Anything that draws, clears, copies, dispatches, writes, waits, orders or signals ends the
// scan (the pass ends as before). The same targets set again keep the open pass too. A draw sampling an attachment of the pass is served as before (same-pass feedback, fb_prepare)
static int rt_hold_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_RT_HOLD"); v = !(e && !strcmp(e, "0")); if (!v) LOG("DM_VK_RT_HOLD=0: every SetRenderTargets ends the open pass"); } return v; }
static int view_eq(uint8_t ha, const view_t *a, uint8_t hb, const view_t *b) { return ha == hb && (!ha || !memcmp(a, b, sizeof *a)); }
static int rt_same(const p_set_rt *a, const p_set_rt *b) {   // the same attachments: count, textures (by serial) and views
    if (!a || !b || a->n != b->n || !a->depth != !b->depth || (a->depth && a->depth->serial != b->depth->serial) || !view_eq(a->has_dview, &a->dview, b->has_dview, &b->dview)) return 0;
    for (int k = 0; k < a->n && k < 8; k++) { if (!a->color[k] != !b->color[k] || (a->color[k] && a->color[k]->serial != b->color[k]->serial)) return 0;
                                              if (a->color[k] && !view_eq(a->has_cview[k], &a->cview[k], b->has_cview[k], &b->cview[k])) return 0; }
    return 1;
}
static const uint8_t *g_rs[32]; static int g_nrs;   // the replay's return points: after each CALL being replayed (mtl_exec_)
static const uint64_t *g_rest; static int g_nrest;   // the submission's command handles after the one being replayed (mtl_exec_rest)
static const uint8_t *rt_hold_scan(const uint8_t *p, const p_set_rt *open) {   // p: the op after the detour's SetRenderTargets; the SetRenderTargets back to `open`, or NULL
    if (g_nrs > 32) return NULL;   // calls nested deeper than the return points kept
    const uint8_t *stk[16]; int ns = 0, outer = g_nrs, ri = 0;
    for (int guard = 0; guard < 4096; guard++) {
        op_hdr h; memcpy(&h, p, sizeof h); const uint8_t *a = p + 8;
        switch (h.op) {
            case OP_END:
                if (ns) { p = stk[--ns]; continue; }
                if (outer > 0) { p = g_rs[--outer]; continue; }
                if (ri < g_nrest && g_rest && g_rest[ri]) { p = (const uint8_t *)(uintptr_t)g_rest[ri++]; continue; }
                return NULL;
            case OP_JUMP: memcpy(&p, a, 8); continue;
            case OP_CALL: { const uint8_t *t; memcpy(&t, a, 8); if (ns >= 16) return NULL; stk[ns++] = p + h.size; p = t; continue; }
            case OP_SET_RT: if (rt_same(open, (const p_set_rt *)a)) return p; break;   // another detour target: still only state
            case OP_BEGIN: case OP_VIEWPORTS: case OP_SCISSOR: case OP_DEPTH_RANGE: case OP_BIND_PROGRAM: case OP_BIND_STATE: case OP_BIND_VATTR: case OP_BIND_VSTREAM: case OP_BIND_VBUF:
            case OP_BIND_UBO: case OP_BIND_SSBO: case OP_BIND_TEX: case OP_BIND_SAMP: case OP_STENCIL: case OP_POLY_OFFSET: case OP_SET_TEX_POOL: case OP_SET_SAMP_POOL: case OP_COPY_STRIDE:
            case OP_BLEND_COLOR: case OP_DEPTH_CLAMP: case OP_VP_SWIZZLE: case OP_BIND_TEX1: case OP_PRIM_RESTART: case OP_BIND_UBO_SMALL: case OP_BIND_VBUF1: case OP_ALPHA_REF: case OP_LINE_WIDTH: break;
            default: return NULL;
        }
        p += h.size;
    }
    return NULL;
}
// DM_VK_DISPATCH_DEFER: the barrier Unreal puts between a light's draw and the volume injection's dispatch (and the one back) ended the pass before dispatch() could defer
// anything (vk19.14's castle run: 0 deferred). A barrier in an open pass is kept from ending it when the next work, past state changes and uniform updates, is a dispatch, or a
// draw while dispatches are deferred; the pass then counts as merged, so a draw sampling one of its written attachments still ends it first (draw_reads_rt), what a deferred
// dispatch writes ends it for a draw that reads it (DRAWQ), and anything else -- other targets, a copy, a clear outside the pass -- ends it as before
static uint64_t g_nbar_kept;
static int next_work_op(const uint8_t *p) {   // the first op from p that is not a state change, a uniform update or a barrier (following calls and jumps as rt_hold_scan does); -1: unknown
    if (g_nrs > 32) return -1;
    const uint8_t *stk[16]; int ns = 0, outer = g_nrs, ri = 0;
    for (int guard = 0; guard < 4096; guard++) {
        op_hdr h; memcpy(&h, p, sizeof h); const uint8_t *a = p + 8;
        switch (h.op) {
            case OP_END:
                if (ns) { p = stk[--ns]; continue; }
                if (outer > 0) { p = g_rs[--outer]; continue; }
                if (ri < g_nrest && g_rest && g_rest[ri]) { p = (const uint8_t *)(uintptr_t)g_rest[ri++]; continue; }
                return -1;
            case OP_JUMP: memcpy(&p, a, 8); continue;
            case OP_CALL: { const uint8_t *t; memcpy(&t, a, 8); if (ns >= 16) return -1; stk[ns++] = p + h.size; p = t; continue; }
            case OP_BEGIN: case OP_VIEWPORTS: case OP_SCISSOR: case OP_DEPTH_RANGE: case OP_BIND_PROGRAM: case OP_BIND_STATE: case OP_BIND_VATTR: case OP_BIND_VSTREAM: case OP_BIND_VBUF:
            case OP_BIND_UBO: case OP_BIND_SSBO: case OP_BIND_TEX: case OP_BIND_SAMP: case OP_STENCIL: case OP_POLY_OFFSET: case OP_SET_TEX_POOL: case OP_SET_SAMP_POOL: case OP_COPY_STRIDE:
            case OP_BLEND_COLOR: case OP_DEPTH_CLAMP: case OP_VP_SWIZZLE: case OP_BIND_TEX1: case OP_PRIM_RESTART: case OP_BIND_UBO_SMALL: case OP_BIND_VBUF1: case OP_ALPHA_REF: case OP_LINE_WIDTH:
            case OP_BARRIER: case OP_UPDATE_UB: break;
            default: return (int)h.op;
        }
        p += h.size;
    }
    return -1;
}
// DM_VK_BARRIER_HOLD (default 1): a barrier between draws into the same targets keeps the pass open too, like the SetRenderTargets detour hold. Unreal's render graph records one
// between consecutive passes (FINAL FANTASY RESONANCE's deferred lights: 5.5 one-draw passes a frame into the scene colour and depth, each loading and storing both). What it orders
// was written by passes already ended (their end barrier covers it) or by dispatches (theirs), so only the open pass's own attachments are at stake: the pass counts as merged
// and a draw sampling one it wrote ends it first (draw_reads_rt). DM_VK_BARRIER_HOLD=0: every barrier ends the open pass, as before
static int rt_split_on(void);
static int barrier_hold_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_BARRIER_HOLD"); v = !(e && !strcmp(e, "0")); if (!v) LOG("DM_VK_BARRIER_HOLD=0: every barrier ends the open pass"); } return v; }
static int barrier_keeps_pass(const uint8_t *next) {
    if (!P.open || !rt_split_on() || (!defer_on() && !barrier_hold_on())) return 0;
    int w = next_work_op(next), draw = w == OP_DRAW || w == OP_DRAW_IDX;
    return (barrier_hold_on() && draw) || (defer_on() && (w == OP_DISPATCH || (g_ndefer && draw)));
}
static void set_rt_(const p_set_rt *r);
static void set_rt(const uint8_t *op, const p_set_rt *r, uint32_t size) {
    if (g_rt_hold) {   // inside a held detour: its SetRenderTargets are not replayed; the one back to the open pass's targets ends it
        if (op == g_rt_hold) { g_rt_hold = NULL; P.rt = r; P.merged = 1; }
        return;
    }
    if (P.open && rt_hold_on() && P.rt) {
        if (rt_same(P.rt, r)) { P.rt = r; P.merged = 1; g_nsame++; return; }   // the same targets again: the pass goes on
        const uint8_t *back = rt_hold_scan(op + size, P.rt);
        if (back) { static int logged; g_rt_hold = back; g_nheld++; if (!logged++) LOG("vk: render targets held across a SetRenderTargets detour (first case; DM_VK_RT_HOLD=0 turns it off)"); return; }
    }
    set_rt_(r);
}
// The format a colour attachment is rendered in: the one tex_view made its view in (g_tv_fmt: the NVN view's format where it sets one, the sRGB twin of a UNORM texture
// encoding on store; else the texture's), so a pass's and a pipeline's colour formats are the view's by construction (vk-t13; ffr-perf's review). Passes and pipelines used the texture's format while the view was sRGB: Tomodachi Life renders its last pass into an sRGB view of
// the RGBA8 window texture; Vulkan wants the pipeline's format to match the view, and the Adreno stored linear values (the whole picture dark and contrasty, the Metal
// host's texture sRGB). DM_VK_RT_VIEW_FMT=0: the texture's format, as before
static int rt_view_fmt_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_RT_VIEW_FMT"); v = !(e && atoi(e) == 0); LOG("vk: DM_VK_RT_VIEW_FMT=%d: colour attachments in their %s format", v, v ? "view's" : "texture's"); } return v; }
static uint64_t g_rt_fmt_differs;   // colour attachments set whose view format is not the texture's (the GPU line)
static void set_rt_(const p_set_rt *r) {
    pass_end();
    memset(&P, 0, sizeof P);
    P.n = r->n > 8 ? 8 : r->n; P.w = P.h = 0x7fffffff; P.rt = r;
    for (int i = 0; i < P.n; i++) {
        if (!r->color[i]) continue;
        vtex *v = tex_get(r->color[i], 1); if (!v) continue;
        P.ct[i] = v; P.cv[i] = tex_view(v, r->color[i], &r->cview[i], r->has_cview[i], -1, 1);
        P.cf[i] = !P.cv[i] ? VK_FORMAT_UNDEFINED : rt_view_fmt_on() ? g_tv_fmt : v->f.f; if (P.cf[i] && P.cf[i] != v->f.f) g_rt_fmt_differs++;
        uint32_t lv = r->has_cview[i] && r->cview[i].set_levels ? r->cview[i].base_level : 0, w = v->w >> lv ? v->w >> lv : 1, h = v->h >> lv ? v->h >> lv : 1;
        P.clv[i] = lv; P.cly[i] = r->has_cview[i] && r->cview[i].set_layers ? r->cview[i].min_layer : 0; P.cwr[i] = 1;
        if (w < P.w) P.w = w; if (h < P.h) P.h = h;
    }
    if (r->depth) {
        vtex *v = tex_get(r->depth, 1);
        if (v && v->f.depth) { P.dt = v; P.dv = tex_view(v, r->depth, &r->dview, r->has_dview, -1, 1); uint32_t lv = r->has_dview && r->dview.set_levels ? r->dview.base_level : 0;
                               if ((v->w >> lv) < P.w) P.w = v->w >> lv ? v->w >> lv : 1; if ((v->h >> lv) < P.h) P.h = v->h >> lv ? v->h >> lv : 1; }
    }
    if (P.w == 0x7fffffff) P.w = P.h = 1;
}

// Pipelines, cached by everything they are built from
typedef struct {
    shader_t *vs, *fs, *gs; int prim;
    va_st attr[16]; int nattr; vs_st stream[16]; int nstream;
    blend_st blend[8]; color_st color; chmask_st chmask; ds_st ds; poly_st poly; int samples;
    VkFormat cf[8]; int nc; VkFormat df; int stencil;
    float line_w; int restart, dual;   // line_w: line primitives only (0 otherwise); dual: blends with the second source, through the pixel shader's mod_dual
} pso_key;
typedef struct { pso_key k; uint64_t h; VkPipeline p; VkPipelineLayout pl; _Atomic int state; uint8_t rewarm; } pso_ent;   // state: 0 being compiled (pso_worker), 1 ready (p may be NULL: failed), 3 warm: made once by the pre-warm and destroyed (DM_VK_PREWARM_KEEP=0), made again from the driver's cache at its first use; rewarm: being made again so
static int g_npso; static _Atomic uint64_t g_pso_ns;   // pipelines made, and the time spent making them (mtl_create_ns: the long-frame lines)
static uint64_t fnv(const void *p, size_t n) { const uint8_t *b = p; uint64_t h = 0xcbf29ce484222325ull; while (n--) h = (h ^ *b++) * 0x100000001b3ull; return h; }
static VkBlendFactor bfac(int f) {
    switch (f) {
        case 1: return VK_BLEND_FACTOR_ZERO; case 2: return VK_BLEND_FACTOR_ONE; case 3: return VK_BLEND_FACTOR_SRC_COLOR; case 4: return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
        case 5: return VK_BLEND_FACTOR_SRC_ALPHA; case 6: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA; case 7: return VK_BLEND_FACTOR_DST_ALPHA; case 8: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
        case 9: return VK_BLEND_FACTOR_DST_COLOR; case 10: return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR; case 11: return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
        case 0x61: return VK_BLEND_FACTOR_CONSTANT_COLOR; case 0x62: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR; case 0x63: return VK_BLEND_FACTOR_CONSTANT_ALPHA; case 0x64: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
        case 0x10: return VK_BLEND_FACTOR_SRC1_COLOR; case 0x11: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR; case 0x12: return VK_BLEND_FACTOR_SRC1_ALPHA; case 0x13: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
        default: return VK_BLEND_FACTOR_ONE;
    }
}
static VkBlendOp bop(int e) { switch (e) { case 2: return VK_BLEND_OP_SUBTRACT; case 3: return VK_BLEND_OP_REVERSE_SUBTRACT; case 4: return VK_BLEND_OP_MIN; case 5: return VK_BLEND_OP_MAX; } return VK_BLEND_OP_ADD; }
static VkStencilOp sop(int o) { return o >= 1 && o <= 8 ? (VkStencilOp)(o - 1) : VK_STENCIL_OP_KEEP; }
static VkCompareOp cop(int f) { return f >= 1 && f <= 8 ? (VkCompareOp)(f - 1) : VK_COMPARE_OP_ALWAYS; }
static int bf_src1(int f) { return f >= 0x10 && f <= 0x13; }   // NVN_BLEND_FUNC_SRC1_COLOR … ONE_MINUS_SRC1_ALPHA
// The second source without the GPU feature (Mali), or without the pixel shader's output 1: read as one (ONE_MINUS_*: zero), as before
static int bf_no_src1(int f) { if (!bf_src1(f)) return f; RED_ONCE(10, "dual-source blending: not available here, the second source reads as one"); return f == 0x11 || f == 0x13 ? 1 : 2; }
// DM_VK_ZERO_ATTR (vk-t9b, default 1): a vertex attribute the vertex shader reads but the game set to format 0 (off) reads zeros from binding ZB (16, a stride-0 binding of
// g_zero_buf), as the Metal host does (nvn_mtl.m ZERO_ATTR). It used to be left out of the pipeline, its input undefined ("[RED] vertex attribute 2 format 0 has no Vulkan
// format", Tomodachi Life's island). Needs 17 vertex bindings (the Adreno has 32); DM_VK_ZERO_ATTR=0: left out as before. g_zattr_draws: the per-600 vertex line
enum { ZB = 16 };
static uint64_t g_zattr_draws;
static int zero_attr_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_ZERO_ATTR"); v = !(e && atoi(e) == 0) && g_props.limits.maxVertexInputBindings > ZB;
    LOG("vk: DM_VK_ZERO_ATTR=%d: vertex attributes of format 0 %s", v, v ? "read zeros (binding 16)" : "left out"); } return v; }
static VkPipeline pso_create(const pso_key *k, VkPipelineLayout pl) {
    VkShaderModule vsm = k->gs ? k->vs->mod : vs_pad_mod(k->vs, k->fs);   // padded where the pixel shader reads what the vertex shader does not write (with a geometry stage between, as it was)
    VkPipelineShaderStageCreateInfo st[3] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_VERTEX_BIT, vsm, "main", NULL}}; uint32_t nst = 1;
    if (k->gs) st[nst++] = (VkPipelineShaderStageCreateInfo){VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_GEOMETRY_BIT, k->gs->mod, "main", NULL};
    if (k->fs) st[nst++] = (VkPipelineShaderStageCreateInfo){VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_FRAGMENT_BIT, k->dual ? k->fs->mod_dual : k->fs->mod, "main", NULL};
    VkVertexInputBindingDescription vb[17]; VkVertexInputAttributeDescription va[16]; uint32_t nvb = 0, nva = 0; uint32_t used = 0; int zused = 0;
    for (int i = 0; i < 16; i++) {
        if (!(k->vs->vin_mask >> i & 1)) continue;
        uint32_t afmt = i < k->nattr ? k->attr[i].fmt : 0;   // an input past the attributes the game set: off, as format 0
        if (!afmt && i >= k->nattr && !zero_attr_on()) continue;
        VkFormat f = afmt ? vfmt(afmt) : VK_FORMAT_UNDEFINED;
        if (!f && zero_attr_on() && !afmt) {   // format 0: the game left the attribute off; it reads zeros, as nvn_mtl.m's ZERO_ATTR (the zero buffer at binding ZB)
            static const VkFormat zf[3] = {VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R32G32B32A32_UINT, VK_FORMAT_R32G32B32A32_SINT};
            va[nva++] = (VkVertexInputAttributeDescription){(uint32_t)i, ZB, zf[k->vs->vin[i] < 3 ? k->vs->vin[i] : 0], 0}; zused = 1; continue; }
        if (!f) { static int n; if (n++ < 8) LOG("[RED] vk: vertex attribute %d format %#x has no Vulkan format", i, k->attr[i].fmt); continue; }
        va[nva++] = (VkVertexInputAttributeDescription){(uint32_t)i, k->attr[i].stream, f, k->attr[i].offset};
        used |= 1u << k->attr[i].stream;
    }
    for (int s = 0; s < 16; s++) if (used >> s & 1) {
        uint32_t stride = s < k->nstream ? k->stream[s].stride : 0, div = s < k->nstream ? k->stream[s].divisor : 0;
        if (div > 1) RED_ONCE(11, "instance divisor %u: drawn as 1", div);
        vb[nvb++] = (VkVertexInputBindingDescription){(uint32_t)s, stride, div ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX};
    }
    if (zused) vb[nvb++] = (VkVertexInputBindingDescription){ZB, 0, VK_VERTEX_INPUT_RATE_VERTEX};   // stride 0: every vertex reads the same 16 zero bytes
    VkPipelineVertexInputStateCreateInfo vi = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, NULL, 0, nvb, vb, nva, va};
    static const VkPrimitiveTopology TOPO[] = {VK_PRIMITIVE_TOPOLOGY_POINT_LIST, VK_PRIMITIVE_TOPOLOGY_LINE_LIST, VK_PRIMITIVE_TOPOLOGY_LINE_STRIP, VK_PRIMITIVE_TOPOLOGY_LINE_STRIP,
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN, VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY, VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY,
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY, VK_PRIMITIVE_TOPOLOGY_PATCH_LIST};
    VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, NULL, 0, TOPO[k->prim < 15 ? k->prim : 4], k->restart ? VK_TRUE : VK_FALSE};
    VkPipelineViewportStateCreateInfo vp = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, NULL, 0, 1, NULL, 1, NULL};
    VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = (VkCullModeFlags)k->poly.cull; rs.frontFace = k->poly.front ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE;
    if (g_origin_ll) rs.frontFace = rs.frontFace == VK_FRONT_FACE_CLOCKWISE ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE;
    rs.depthBiasEnable = k->poly.offset_en != 0; rs.lineWidth = k->line_w > 1 ? k->line_w : 1;
    VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, NULL, 0, (VkSampleCountFlagBits)(k->samples > 1 ? k->samples : 1)};
    VkPipelineDepthStencilStateCreateInfo ds = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    if (k->df) {
        ds.depthTestEnable = k->ds.dt; ds.depthWriteEnable = k->ds.dw; ds.depthCompareOp = k->ds.dt ? cop(k->ds.df) : VK_COMPARE_OP_ALWAYS;
        if (k->stencil && k->ds.st) {
            ds.stencilTestEnable = VK_TRUE;
            ds.front = (VkStencilOpState){sop(k->ds.fail_f), sop(k->ds.zpass_f), sop(k->ds.zfail_f), cop(k->ds.sf_f), 0xff, 0xff, 0};
            ds.back = (VkStencilOpState){sop(k->ds.fail_b), sop(k->ds.zpass_b), sop(k->ds.zfail_b), cop(k->ds.sf_b), 0xff, 0xff, 0};
        }
    }
    VkPipelineColorBlendAttachmentState ba[8];
    for (int i = 0; i < k->nc; i++) {
        const blend_st *b = &k->blend[i]; int en = k->color.blend_en >> i & 1;
        int sr = b->src_rgb, dr = b->dst_rgb, sa = b->src_a, da = b->dst_a;
        if (!k->dual) { sr = bf_no_src1(sr); dr = bf_no_src1(dr); sa = bf_no_src1(sa); da = bf_no_src1(da); }
        ba[i] = (VkPipelineColorBlendAttachmentState){en, bfac(sr), bfac(dr), bop(b->eq_rgb), bfac(sa), bfac(da), bop(b->eq_a), (k->chmask.mask >> (4 * i)) & 15};
    }
    VkPipelineColorBlendStateCreateInfo cb = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, NULL, 0, VK_FALSE, VK_LOGIC_OP_COPY, (uint32_t)k->nc, ba, {0}};
    VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_DEPTH_BIAS, VK_DYNAMIC_STATE_BLEND_CONSTANTS,
                            VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK, VK_DYNAMIC_STATE_STENCIL_WRITE_MASK, VK_DYNAMIC_STATE_STENCIL_REFERENCE};
    VkPipelineDynamicStateCreateInfo dy = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, NULL, 0, sizeof dyn / sizeof *dyn, dyn};
    VkPipelineRenderingCreateInfo rinfo = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, NULL, 0, (uint32_t)k->nc, k->cf, k->df, k->stencil ? k->df : VK_FORMAT_UNDEFINED};
    VkRenderPass crp = VK_NULL_HANDLE;
    if (!g_dynrend) { VkAttachmentLoadOp ld[8]; for (int i = 0; i < 8; i++) ld[i] = VK_ATTACHMENT_LOAD_OP_LOAD;
                      crp = rp_get(k->cf, ld, k->nc, k->df, k->stencil, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_LOAD_OP_LOAD, VK_IMAGE_LAYOUT_GENERAL); }
    VkGraphicsPipelineCreateInfo gi = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, g_dynrend ? &rinfo : NULL, 0, nst, st, &vi, &ia, NULL, &vp, &rs, &ms, &ds, &cb, &dy, pl, crp, 0, VK_NULL_HANDLE, -1};
    uint64_t t0 = now_ns();
    VkPipeline p; VkResult r = vkCreateGraphicsPipelines(g_dev, g_pcache, 1, &gi, NULL, &p);
    atomic_fetch_add(&g_pso_ns, now_ns() - t0);
    if (r != VK_SUCCESS) {
        LOG("[RED] vk: pipeline for %s / %s not created (%d)", k->vs->key, k->fs ? k->fs->key : "-", (int)r); p = VK_NULL_HANDLE;
        // What the driver refused (Tomodachi Life's c0b81b887fb0137d / d317a98e994ca9dd: -13 on the Adreno, both modules valid SPIR-V): the state, then the same pipeline without the
        // pixel shader, which tells a fragment-stage refusal from the rest. The first 4 failures only
        static _Atomic int nd; if (atomic_fetch_add(&nd, 1) < 4) {
            char ln[768]; int o = snprintf(ln, sizeof ln, "vk: refused pipeline: stages %u%s prim %d topology %d restart %d, vs inputs %#x, %u attributes", nst, k->gs ? " (geometry)" : "", k->prim, (int)ia.topology, k->restart, k->vs->vin_mask, nva);
            for (uint32_t i = 0; i < nva && o < (int)sizeof ln - 40; i++) o += snprintf(ln + o, sizeof ln - o, " [loc %u bind %u fmt %d off %u]", va[i].location, va[i].binding, (int)va[i].format, va[i].offset);
            o += snprintf(ln + o, sizeof ln - o, ", %d targets", k->nc);
            for (int i = 0; i < k->nc && o < (int)sizeof ln - 24; i++) o += snprintf(ln + o, sizeof ln - o, " fmt %d blend %d", (int)k->cf[i], (int)ba[i].blendEnable);
            snprintf(ln + o, sizeof ln - o, ", depth fmt %d stencil %d, samples %d, dual %d, line %g, dynamic rendering %d", (int)k->df, k->stencil, k->samples, k->dual, k->line_w, g_dynrend);
            LOG("%s", ln);
            if (k->fs && nst > 1) { VkGraphicsPipelineCreateInfo g2 = gi; g2.stageCount = nst - 1; VkPipeline p2 = VK_NULL_HANDLE;
                VkResult r2 = vkCreateGraphicsPipelines(g_dev, VK_NULL_HANDLE, 1, &g2, NULL, &p2);
                LOG("vk: refused pipeline without its pixel shader: %d (%s)", (int)r2, r2 == VK_SUCCESS ? "the pixel shader is what the driver refuses" : "refused without it too");
                if (p2) vkDestroyPipeline(g_dev, p2, NULL); }
        }
    }
    return p;
}
// Pipelines by their key: a hash table of entries (each allocated once, so a worker can fill one in while the table grows). A new one is made on a worker thread (pso_worker) when
// may_wait is 0 -- the draw is left out until it is ready, a few frames -- else on the spot as before. Effects meeting the GPU for the first time compiled dozens at once on the
// replay thread: FINAL FANTASY RESONANCE froze 0.5-1.8 s in battle (42 new pipelines in one frame). Only draws in passes with colour and depth (the scene, its effects) wait for
// a worker; passes drawn once (the colour grading LUT, render-to-texture, post-processing, shadows) still make theirs at once
static pso_ent **g_psot; static uint32_t g_psot_cap; static _Atomic uint64_t g_pso_async, g_pso_skipped; static _Atomic int g_pso_pending;
static uint64_t g_pso_last_new;   // the frame of the last new pipeline (pcache_save soon after)
static pthread_mutex_t g_psoq_mu = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t g_psoq_cv = PTHREAD_COND_INITIALIZER; static pso_ent **g_psoq; static int g_npsoq, g_cpsoq;
static _Atomic uint64_t g_pso_rewarm, g_pso_rewarm_ns;   // warm pipelines made again at their first use, and the time that took (DM_VK_PREWARM_KEEP=0)
static void *pso_worker(void *a) { pthread_setname_np("dm-pso");
    (void)a;
    for (;;) {
        pthread_mutex_lock(&g_psoq_mu); while (!g_npsoq) pthread_cond_wait(&g_psoq_cv, &g_psoq_mu);
        pso_ent *e = g_psoq[0]; memmove(g_psoq, g_psoq + 1, --g_npsoq * sizeof *g_psoq); pthread_mutex_unlock(&g_psoq_mu);
        uint64_t t0 = now_ns(); e->p = pso_create(&e->k, e->pl); if (e->rewarm) atomic_fetch_add(&g_pso_rewarm_ns, now_ns() - t0);
        atomic_store_explicit(&e->state, 1, memory_order_release); atomic_fetch_sub(&g_pso_pending, 1);
    }
    return NULL;
}
static void pso_enqueue(pso_ent *e) {
    pthread_mutex_lock(&g_psoq_mu);
    if (g_npsoq == g_cpsoq) { g_cpsoq = g_cpsoq ? 2 * g_cpsoq : 64; g_psoq = realloc(g_psoq, g_cpsoq * sizeof *g_psoq); }
    g_psoq[g_npsoq++] = e; atomic_fetch_add(&g_pso_pending, 1); atomic_fetch_add(&g_pso_async, 1);
    pthread_cond_signal(&g_psoq_cv); pthread_mutex_unlock(&g_psoq_mu);
}
// Every pipeline made is recorded (pipeline_keys.bin beside the pipeline cache: its shaders' keys and the rest of its key), and the next run makes them all again on a thread of its
// own after its first frame (pso_precreate), so an effect or a skill met once does not stall the game when it comes back, also when the driver's cache dropped the pipeline
typedef struct { char vs[17], fs[17], gs[17]; pso_key k; } pso_rec;
static pthread_mutex_t g_psomu = PTHREAD_MUTEX_INITIALIZER; static FILE *g_pso_rec;
static void pso_record(const pso_key *k) {
    if (!g_pso_rec) { const char *d = getenv("DM_VK_DIR"); if (!d) return; char p[1100]; snprintf(p, sizeof p, "%s/../pipeline_keys.bin", d); if (!(g_pso_rec = fopen(p, "ab"))) return; }
    pso_rec r; memset(&r, 0, sizeof r); r.k = *k; r.k.vs = r.k.fs = r.k.gs = NULL;
    if (k->vs) strcpy(r.vs, k->vs->key); if (k->fs) strcpy(r.fs, k->fs->key); if (k->gs) strcpy(r.gs, k->gs->key);
    fwrite(&r, sizeof r, 1, g_pso_rec); fflush(g_pso_rec);
}
static pso_ent *pso_find_(const pso_key *k, uint64_t h, uint32_t *slot) {   // under g_psomu
    if (!g_psot_cap) { g_psot_cap = 4096; g_psot = calloc(g_psot_cap, sizeof *g_psot);
                       for (int i = 0; i < 2; i++) { pthread_t t; if (!pthread_create(&t, NULL, pso_worker, NULL)) pthread_detach(t); } }
    uint32_t i = (uint32_t)h & (g_psot_cap - 1);
    for (; g_psot[i]; i = (i + 1) & (g_psot_cap - 1)) if (g_psot[i]->h == h && !memcmp(&g_psot[i]->k, k, sizeof *k)) return g_psot[i];
    *slot = i; return NULL;
}
static pso_ent *pso_insert_(const pso_key *k, uint64_t h, VkPipelineLayout pl, uint32_t i) {   // under g_psomu
    pso_ent *e = calloc(1, sizeof *e); e->k = *k; e->h = h; e->pl = pl;
    g_psot[i] = e; g_npso++;
    if ((uint32_t)g_npso * 2 > g_psot_cap) {   // grow the table (entries stay where they are)
        uint32_t nc = g_psot_cap * 2; pso_ent **nt = calloc(nc, sizeof *nt);
        for (uint32_t j = 0; j < g_psot_cap; j++) if (g_psot[j]) { uint32_t q = (uint32_t)g_psot[j]->h & (nc - 1); while (nt[q]) q = (q + 1) & (nc - 1); nt[q] = g_psot[j]; }
        free(g_psot); g_psot = nt; g_psot_cap = nc;
    }
    return e;
}
static uint64_t g_pso_look[4];   // pipeline lookups: 0 found ready, 1 found warm (made again), 2 found being made, 3 not found (new) -- the pipelines line, to see whether the pre-warm's entries are met
static VkPipeline pso_get(const pso_key *k, VkPipelineLayout pl, int may_wait) {
    uint64_t h = fnv(k, sizeof *k); uint32_t i = 0;
    pthread_mutex_lock(&g_psomu);
    pso_ent *e = pso_find_(k, h, &i);
    if (e) {
        pthread_mutex_unlock(&g_psomu);
        { int st0 = atomic_load_explicit(&e->state, memory_order_acquire); g_pso_look[st0 == 1 ? 0 : st0 == 3 ? 1 : 2]++; }
        for (;;) {
            int st = atomic_load_explicit(&e->state, memory_order_acquire);
            if (st == 1) return e->p;
            if (st == 3) {   // warm (made by the pre-warm, then destroyed): made again now, from the driver's cache
                int exp = 3; if (!atomic_compare_exchange_strong(&e->state, &exp, 0)) continue;
                e->rewarm = 1; atomic_fetch_add(&g_pso_rewarm, 1);
                if (may_wait) { uint64_t t0 = now_ns(); e->p = pso_create(k, pl); atomic_fetch_add(&g_pso_rewarm_ns, now_ns() - t0); atomic_store_explicit(&e->state, 1, memory_order_release); return e->p; }
                pso_enqueue(e); atomic_fetch_add(&g_pso_skipped, 1); return VK_NULL_HANDLE;
            }
            if (may_wait) { usleep(200); continue; }   // a worker has it: wait for that one
            atomic_fetch_add(&g_pso_skipped, 1); return VK_NULL_HANDLE;
        }
    }
    e = pso_insert_(k, h, pl, i); pthread_mutex_unlock(&g_psomu); g_pso_look[3]++;
    g_pso_last_new = atomic_load(&g_presents); pso_record(k);
    if (may_wait) { e->p = pso_create(k, pl); atomic_store_explicit(&e->state, 1, memory_order_release); return e->p; }
    pso_enqueue(e);
    atomic_fetch_add(&g_pso_skipped, 1); return VK_NULL_HANDLE;
}

static int hexkey(const char *s, uint8_t md[8]) { for (int i = 0; i < 8; i++) if (sscanf(s + 2 * i, "%2hhx", &md[i]) != 1) return 0; return 1; }
static _Atomic uint64_t g_pre_made, g_pre_had, g_pre_slow;
// DM_VK_PREWARM_KEEP=0: the pre-warm makes each pipeline and destroys it at once, so only the driver's pipeline cache is warm; a pipeline is made again (from that cache) at its first
// use. FINAL FANTASY RESONANCE: 6619 pipelines kept alive cost RSS +485 MB at boot on a tablet already swapping 2 GB; with 0 the pre-warm took 1.6 s instead of 22.6 s, RSS +65 MB,
// the game's swap 0.9 GB lower, and a pipeline made again from the cache took ~0.4 ms. Default 0 since vk19.13; DM_VK_PREWARM_KEEP=1 keeps them all alive as before
static int prewarm_keep(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_PREWARM_KEEP"); v = e && atoi(e) != 0; LOG("DM_VK_PREWARM_KEEP=%d: %s", v, v ? "the pre-warm keeps every pipeline alive" : "the pre-warm keeps only the driver's pipeline cache warm"); } return v; }
static void pso_precreate_one(const pso_rec *r) {
    uint8_t md[8]; pso_key k = r->k;
    if (!r->vs[0] || !hexkey(r->vs, md) || !(k.vs = shader_get(md)) || !k.vs->mod) return;
    if (r->fs[0]) { if (!hexkey(r->fs, md) || !(k.fs = shader_get(md)) || !k.fs->mod) return; }
    if (r->gs[0]) { if (!hexkey(r->gs, md) || !(k.gs = shader_get(md)) || !k.gs->mod) return; }
    if (k.dual && (!k.fs || !fs_dual_mod(k.fs))) return;
    VkPipelineLayout pl = layout_get(k.vs, k.fs, k.gs);
    uint64_t h = fnv(&k, sizeof k); uint32_t i = 0;
    pthread_mutex_lock(&g_psomu);
    if (pso_find_(&k, h, &i)) { pthread_mutex_unlock(&g_psomu); atomic_fetch_add(&g_pre_had, 1); return; }
    pso_ent *e = pso_insert_(&k, h, pl, i); pthread_mutex_unlock(&g_psomu);
    uint64_t t0 = now_ns(); e->p = pso_create(&k, pl); atomic_fetch_add(&g_pre_made, 1);
    if (now_ns() - t0 > 3000000ull) atomic_fetch_add(&g_pre_slow, 1);   // not in the driver's cache: compiled
    if (!prewarm_keep() && e->p) { vkDestroyPipeline(g_dev, e->p, NULL); e->p = VK_NULL_HANDLE; atomic_store_explicit(&e->state, 3, memory_order_release); }   // the driver's cache holds it now
    else atomic_store_explicit(&e->state, 1, memory_order_release);
}
static int g_prewarm_ran;
static int g_prewarm_depth = -1;   // the depth mode the pre-warm loaded shaders under (dm_vk_prewarm), from the last run's vk_depth_mode
static void *pso_precreate(void *a) { pthread_setname_np("dm-psopre");
    (void)a; const char *d = getenv("DM_VK_DIR"); if (!d || g_prewarm_ran) return NULL;   // the pre-warm before the game did it already
    char p[1100]; snprintf(p, sizeof p, "%s/../pipeline_keys.bin", d); FILE *f = fopen(p, "rb"); if (!f) return NULL;
    uint64_t t0 = now_ns(); pso_rec r; uint64_t n = 0;
    while (fread(&r, sizeof r, 1, f) == 1) { n++; pso_precreate_one(&r); }
    fclose(f);
    LOG("vk: pipelines of earlier runs made again: %llu of %llu recorded (%llu already there) in %.1f s", (unsigned long long)atomic_load(&g_pre_made), (unsigned long long)n,
        (unsigned long long)atomic_load(&g_pre_had), (now_ns() - t0) / 1e9);
    return NULL;
}
// ---------- Draw resources ----------
// ---------- Run-time switches (vk19.59, improvement plan v4) ----------
// The new switches of the plan's builds live in one table: each is read from the environment once, and the ones on the run-time list may also be set while the game runs
// from DM_VK_CTL_FILE ("set <name> <value>", vk_ctl: the file is read again only when its modification time changes, at most once a frame). The value is an atomic; the
// recording thread takes a copy of every value once per submission (tog_latch at mtl_sub_begin) and reads only that copy (tog), so a switch never changes in the middle
// of a submission. A name not on the run-time list is refused when the file sets it (it is read at start only)
enum { TG_SHADOW, TG_IDX_NEON, TG_FB_FAST, TG_LEAN, TG_PACE, TG_DC2, TG_TSTORE, TG_BSTORE, TG_N };
// The range stores (TG_TSTORE, TG_BSTORE) are per game (merge-main): on by default only where the game profile's range_store key names them (game.py: FINAL FANTASY RESONANCE
// sets "texel buf"), because they write-protect guest pages (nvn_dirty), measured on that game only; FINAL FANTASY XII's profile records a hang from write-protected textures
// (tex_dirty_upload). The variables still override both ways (0 / 1 / 2). tog_init_ logs which decided: the profile, the environment, or the default (off)
#if !defined(DM_GAME_RANGE_STORE_TEXEL) || !defined(DM_GAME_RANGE_STORE_BUF)
#error "game_profile.h has no range_store macros: regenerate it with dm/games/game.py"
#endif
static struct { const char *name; int runtime; _Atomic int val; int def; } g_tog[TG_N] = {   // def: the value without the variable (on by default after its A/B; =0 turns it off)
    [TG_SHADOW] = {"DM_VK_SHADOW", 1},   // diagnostic: what a persistent cache could save (copy_in / index-scan misses whose bytes an earlier copy already had)
    [TG_IDX_NEON] = {"DM_VK_IDX_NEON", 1, 0, 1},   // vk19.60: the index scan with NEON min/max (1), checked against the scalar scan (2); on by default from vk19.65 (session 1: -1.0 ms)
    [TG_PACE] = {"DM_VK_PRESENT_PACE", 0},   // vk19.63: one setFrameRate request for the cap at the first swapchain, FIFO when the cap is at most the panel's rate; start only
    [TG_TSTORE] = {"DM_VK_TEXEL_STORE", 0, 0, DM_GAME_RANGE_STORE_TEXEL},   // vk19.70: uniform texel buffers kept across frames in a store, validated by nvn_dirty (1), checked against a host copy (2); start only; on by default from vk19.73 (session 4: recording -0.6...-0.9, RHIThread -0.6...-0.8 ms)
    [TG_BSTORE] = {"DM_VK_BUF_STORE", 0, 0, DM_GAME_RANGE_STORE_BUF},   // vk19.72: vertex streams and the indirect draws' index pool in the same store (1), checked against a host copy (2); start only; on by default from vk19.75b (session 5: recording -0.2, RHIThread -0.25 ms; vertex only in practice)
    [TG_DC2] = {"DM_VK_DC_WAYS2", 1},   // vk19.69: copy_in's table two-way set-associative (C5); on the run-time list (a mode change only costs misses)
    [TG_LEAN] = {"DM_VK_LEAN", 0, 0, 1},   // vk19.62: small hot-path items (trace pointers, the memo hash, the timer conversion, sampled descriptor-set timing); start only; on by default from vk19.65 (session 1: RHIThread -1.0 ms)
    [TG_FB_FAST] = {"DM_VK_FB_FAST", 1, 0, 1},   // vk19.61: same-pass feedback found by texture serial, a per-shader cache of "samples none" (1), checked against the full way (2); on by default from vk19.65 (session 1: -0.4 ms)
};
static int g_tog_lat[TG_N];
static void tog_init_(void) {
    for (int i = 0; i < TG_N; i++) { const char *e = getenv(g_tog[i].name); int v = e ? atoi(e) : g_tog[i].def; atomic_store(&g_tog[i].val, v);
        if (i == TG_TSTORE || i == TG_BSTORE) { LOG("vk: %s=%d (%s)", g_tog[i].name, v, e ? "env" : g_tog[i].def ? "profile: range_store" : "default off: not in the profile's range_store"); continue; }   // the range stores: always logged, with the source
        if (v) LOG("vk: %s=%d (%s%s)", g_tog[i].name, v, e ? "from the environment" : "the default", g_tog[i].runtime ? "; may be changed at run time" : ""); else if (g_tog[i].def) LOG("vk: %s=0: off (on by default)", g_tog[i].name); }
}
static pthread_once_t g_tog_once = PTHREAD_ONCE_INIT;
static void tog_init(void) { pthread_once(&g_tog_once, tog_init_); }   // vk_init, and the control file's first set (vk19.59c: once, whichever thread)
static void tog_latch(void) { for (int i = 0; i < TG_N; i++) g_tog_lat[i] = atomic_load_explicit(&g_tog[i].val, memory_order_relaxed); g_lean_tr = g_tog_lat[TG_LEAN]; }
static inline int tog(int i) { return g_tog_lat[i]; }
static void tog_set(const char *name, int v, uint64_t frame) {   // the control file (present thread)
    tog_init();
    for (int i = 0; i < TG_N; i++) if (!strcmp(g_tog[i].name, name)) {
        if (!g_tog[i].runtime) { LOG("[RED] vk: control file: %s is not on the run-time list: read at start only (frame %llu)", name, (unsigned long long)frame); return; }
        int old = atomic_exchange(&g_tog[i].val, v); if (old != v) LOG("vk: control file: %s %d -> %d (frame %llu; from the next submission)", name, old, v, (unsigned long long)frame); return; }
    LOG("[RED] vk: control file: no switch %s on the run-time list (frame %llu)", name, (unsigned long long)frame);
}
// ---------- Copy and index-scan counters (vk19.59, plan v4 step 0) ----------
// copy_in per caller (what asked for the copy) x miss reason x the pool's CPU access, the index-range cache, texel buffer views made, tex_get calls; logged per 600 frames.
// The pool's class is looked up on misses only, through a one-pool cache (stats only: a pool deleted meanwhile at most counts one miss in the wrong class)
enum { CI_VTX, CI_IDX, CI_IDXPOOL, CI_UBO, CI_SSBO, CI_TEXEL, CI_IMG, CI_N };
enum { MR_EMPTY, MR_OTHER, MR_GEN, MR_SIZE, MR_ALIGN, MR_WLOG, MR_N };
enum { PC_GPU, PC_CACHED, PC_UNCACHED, PC_NONE, PC_N };
static uint64_t g_ci_hit[CI_N], g_ci_miss[CI_N], g_ci_mbytes[CI_N], g_ci_why[MR_N], g_ci_pool[PC_N];
static uint64_t g_ic_hit, g_ic_scan, g_ic_idx, g_ic_same, g_ic_coll, g_ic_new, g_ic_keys, g_ic_keys_over;
static _Atomic uint64_t g_bv_made;   // texel buffer views made (bview)
static int pool_class(uint64_t addr);
// Shadow mode (DM_VK_SHADOW=1, diagnostic, off by default): each copy_in miss and each index scan hashes the bytes it took and looks the (address, size) up in a table of
// earlier ones: the same bytes copied before in another submission of this frame, or in an earlier frame, are what a cache kept across submissions could have saved (an
// estimate of a persistent cache's reach before building one: the table is direct-mapped, 64K entries, and matches the exact address and length only, so a range pushed
// out of its slot or read at another offset or length is not counted); the same submission counts apart (a better per-submission cache). Off: no cost beyond one branch
enum { NSH = 1 << 16 };
static struct { uint64_t addr, h, sub, frame; uint32_t len; uint8_t kind; } *g_shw;
static uint64_t g_shw_n[2][3], g_shw_b[2][3], g_shw_all[2], g_shw_ns;   // [copy, scan][same submission, another of this frame, an earlier frame]
static uint64_t shadow_hash(const uint8_t *p, uint64_t n) {
    uint64_t h = 0x9e3779b97f4a7c15ull ^ n, x; uint64_t i = 0;
    for (; i + 8 <= n; i += 8) { memcpy(&x, p + i, 8); h = (h ^ x) * 0xff51afd7ed558ccdull; h ^= h >> 32; }
    for (; i < n; i++) h = (h ^ p[i]) * 0x100000001b3ull;
    return h;
}
static uint64_t g_shw_cb[CI_N][2];   // copies' repeated bytes by caller: [another submission of the frame, an earlier frame] (vk19.66m)
static int g_shw_caller = -1;   // the copy_in caller of the shadow_note in progress
static void shadow_note(int kind, uint64_t addr, uint64_t len) {   // kind 0 a copy, 1 an index scan
    if (!tog(TG_SHADOW) || !addr || !len) return;
    uint64_t t0 = now_ns();
    if (!g_shw && !(g_shw = calloc(NSH, sizeof *g_shw))) return;
    uint64_t h = shadow_hash((const uint8_t *)(uintptr_t)addr, len), sub = g_bvgen, fr = atomic_load_explicit(&g_presents, memory_order_relaxed);
    typeof(g_shw[0]) *e = &g_shw[(addr >> 4 ^ addr >> 20 ^ len * 0x9e37u ^ (uint64_t)kind << 15) % NSH];
    g_shw_all[kind]++;
    if (e->addr == addr && e->len == len && e->kind == kind && e->h == h) { int w = e->sub == sub ? 0 : e->frame == fr ? 1 : 2; g_shw_n[kind][w]++; g_shw_b[kind][w] += len;
        if (!kind && w && g_shw_caller >= 0 && g_shw_caller < CI_N) g_shw_cb[g_shw_caller][w - 1] += len; }
    *e = (typeof(*e)){addr, h, sub, fr, (uint32_t)(len > UINT32_MAX ? UINT32_MAX : len), (uint8_t)kind};
    g_shw_ns += now_ns() - t0;
}
// DM_VK_IDX_NEON (vk19.60, plan v4 C1, off by default; on the run-time list): the lowest and highest vertex an index range uses, by NEON min/max over 16 / 8 / 4 lanes,
// the index size chosen once outside the loop (the scalar scan tests it per index through IDX). The restart index (all ones at the index's size) is left out as the scalar
// scan does: it is the largest value, so the minimum over raw lanes is right unless every index is a restart (then the minimum is the restart itself: answered as none), and
// the maximum is taken over lanes with restarts cleared (v & ~(v == restart)), which only lowers them to 0, never above the true maximum. =2 (verify): both scans, the scalar
// result used, differences counted and the first 8 logged
#if defined(__aarch64__)
#include <arm_neon.h>
#endif
static uint64_t g_neon_chk, g_neon_bad;
static void idx_scan_scalar(const void *src, int32_t n, uint32_t isz, uint32_t *plo, uint32_t *phi) {
    uint32_t lo = UINT32_MAX, hi = 0;
    if (isz == 1) { const uint8_t *p = src; for (int32_t i = 0; i < n; i++) { uint32_t v = p[i]; if (v == 0xff) continue; if (v < lo) lo = v; if (v > hi) hi = v; } }
    else if (isz == 2) { const uint16_t *p = src; for (int32_t i = 0; i < n; i++) { uint32_t v = p[i]; if (v == 0xffff) continue; if (v < lo) lo = v; if (v > hi) hi = v; } }
    else { const uint32_t *p = src; for (int32_t i = 0; i < n; i++) { uint32_t v = p[i]; if (v == 0xffffffffu) continue; if (v < lo) lo = v; if (v > hi) hi = v; } }
    *plo = lo; *phi = hi;
}
#if defined(__aarch64__)
static void idx_scan_neon(const void *src, int32_t n, uint32_t isz, uint32_t *plo, uint32_t *phi) {
    uint32_t lo, hi; int32_t i = 0;
    if (isz == 2) {
        const uint16_t *p = src; uint16x8_t mn0 = vdupq_n_u16(0xffff), mn1 = mn0, mx0 = vdupq_n_u16(0), mx1 = mx0; const uint16x8_t r = vdupq_n_u16(0xffff);
        for (; i + 16 <= n; i += 16) { uint16x8_t a = vld1q_u16(p + i), b = vld1q_u16(p + i + 8);
            mn0 = vminq_u16(mn0, a); mn1 = vminq_u16(mn1, b); mx0 = vmaxq_u16(mx0, vbicq_u16(a, vceqq_u16(a, r))); mx1 = vmaxq_u16(mx1, vbicq_u16(b, vceqq_u16(b, r))); }
        lo = vminvq_u16(vminq_u16(mn0, mn1)); hi = vmaxvq_u16(vmaxq_u16(mx0, mx1));
        for (; i < n; i++) { uint32_t v = p[i]; if (v < lo) lo = v; if (v != 0xffff && v > hi) hi = v; }
        if (lo == 0xffff) lo = UINT32_MAX;
    } else if (isz == 4) {
        const uint32_t *p = src; uint32x4_t mn0 = vdupq_n_u32(0xffffffffu), mn1 = mn0, mx0 = vdupq_n_u32(0), mx1 = mx0; const uint32x4_t r = vdupq_n_u32(0xffffffffu);
        for (; i + 8 <= n; i += 8) { uint32x4_t a = vld1q_u32(p + i), b = vld1q_u32(p + i + 4);
            mn0 = vminq_u32(mn0, a); mn1 = vminq_u32(mn1, b); mx0 = vmaxq_u32(mx0, vbicq_u32(a, vceqq_u32(a, r))); mx1 = vmaxq_u32(mx1, vbicq_u32(b, vceqq_u32(b, r))); }
        lo = vminvq_u32(vminq_u32(mn0, mn1)); hi = vmaxvq_u32(vmaxq_u32(mx0, mx1));
        for (; i < n; i++) { uint32_t v = p[i]; if (v < lo) lo = v; if (v != 0xffffffffu && v > hi) hi = v; }
    } else {
        const uint8_t *p = src; uint8x16_t mn = vdupq_n_u8(0xff), mx = vdupq_n_u8(0); const uint8x16_t r = vdupq_n_u8(0xff);
        for (; i + 16 <= n; i += 16) { uint8x16_t a = vld1q_u8(p + i); mn = vminq_u8(mn, a); mx = vmaxq_u8(mx, vbicq_u8(a, vceqq_u8(a, r))); }
        lo = vminvq_u8(mn); hi = vmaxvq_u8(mx);
        for (; i < n; i++) { uint32_t v = p[i]; if (v < lo) lo = v; if (v != 0xff && v > hi) hi = v; }
        if (lo == 0xff) lo = UINT32_MAX;
    }
    if (lo == UINT32_MAX) hi = 0;   // every index a restart: as the scalar scan (none)
    *plo = lo; *phi = hi;
}
#else
static void idx_scan_neon(const void *src, int32_t n, uint32_t isz, uint32_t *plo, uint32_t *phi) { idx_scan_scalar(src, n, isz, plo, phi); }   // no NEON: the scalar scan
#endif
static void idx_scan(const void *src, int32_t n, uint32_t isz, uint32_t *plo, uint32_t *phi) {
    int m = tog(TG_IDX_NEON);
    if (!m) { idx_scan_scalar(src, n, isz, plo, phi); return; }
    idx_scan_neon(src, n, isz, plo, phi);
    if (m == 2) { uint32_t lo, hi; idx_scan_scalar(src, n, isz, &lo, &hi); g_neon_chk++;
        if (lo != *plo || hi != *phi) { g_neon_bad++; static int nl; if (nl++ < 8) LOG("[RED] vk: DM_VK_IDX_NEON=2: %d indices of %u bytes at %p: NEON %u-%u, scalar %u-%u", n, isz, src, *plo, *phi, lo, hi); }
        *plo = lo; *phi = hi; }
}
// Distinct index ranges (address, count, index size) scanned in a 600-frame window: an open-addressed set, emptied at each window's line; full: counted as over
enum { NICK = 1 << 14 };
static struct { uint64_t a; int32_t n; uint8_t isz; } g_ick[NICK];
static void ic_key(uint64_t a, int32_t n, uint32_t isz) {
    uint32_t h = (uint32_t)((a >> 4 ^ a >> 21 ^ (uint64_t)n * 0x9e3779b1u ^ isz) % NICK);
    for (int k = 0; k < 64; k++, h = (h + 1) % NICK) {
        if (g_ick[h].a == a && g_ick[h].n == n && g_ick[h].isz == isz) return;
        if (!g_ick[h].a) { g_ick[h].a = a; g_ick[h].n = n; g_ick[h].isz = (uint8_t)isz; g_ic_keys++; return; } }
    g_ic_keys_over++;
}
typedef struct { uint64_t addr, size, gen; VkDeviceSize off; uint32_t seq; uint8_t cls; } dcopy;   // cls: the range's pool class (PC_*; PC_N not looked up), for the copy counters
#define NDC 4096
static dcopy g_dc[NDC]; static int g_ndc;
// The in-order writes of this generation (wlog_add: inline uniform updates, buffer copies and fills), by range: a cached copy stays good unless one written after it overlaps it.
// Every write used to end every cached copy: Unreal updates uniforms between draws, so each draw copied its vertex streams, uniform and texel buffers again, at new ring
// offsets, with new texel buffer views and new descriptor sets (14% of the sets reused, 12 ms a frame in descriptor sets)
enum { NWLOG = 8192 };
static uint64_t g_wlo[NWLOG], g_whi[NWLOG];
static void wlog_add(uint64_t a, uint64_t n) {
    if (g_nwlog == NWLOG) { g_wgen++; g_nwlog = 0; g_h0_gen[2]++; return; }   // too many: a new generation (every copy taken again)
    g_wlo[g_nwlog] = a; g_whi[g_nwlog] = a + n; g_nwlog++;
}
static VkDeviceSize copy_in(uint64_t addr, uint64_t size, VkDeviceSize align, int caller) {   // guest memory → ring, reused when the same range is asked for again unchanged (hashed by address)
    dcopy *e; int why;
    // DM_VK_DC_WAYS2 (vk19.69, plan C5, off by default): two ways a set (NDC / 2 sets): the way holding this address, else a free or older-submission one, else the one
    // used less recently. Session 1: 37% of copies found the slot held by another range; the same-submission repeats (275 copies, 2.0 MB a frame) are what it can catch
    static uint8_t lru[NDC / 2];   // the way to replace next in each set
    if (tog(TG_DC2)) {
        uint32_t st = (uint32_t)((addr >> 4 ^ addr >> 13) % (NDC / 2)); dcopy *w = &g_dc[2 * st];
        int k = w[0].addr == addr ? 0 : w[1].addr == addr ? 1 : -1;
        if (k < 0) k = w[0].gen != g_wgen ? 0 : w[1].gen != g_wgen ? 1 : lru[st];   // a miss of another range: a free or stale way first
        e = &w[k]; lru[st] = (uint8_t)!k;   // the other way is now the less recently used
    } else e = &g_dc[(addr >> 4 ^ addr >> 13) % NDC];
    if (e->addr == addr && e->size >= size && e->gen == g_wgen && e->gen && e->off % align == 0 && e->seq <= g_nwlog) {
        uint64_t hi = addr + size; uint32_t k = e->seq;
        for (; k < g_nwlog; k++) if (g_wlo[k] < hi && g_whi[k] > addr) break;
        g_h0_wscan[0] += k - e->seq; g_h0_wscan_n[0]++;
        if (k == g_nwlog) { if (size >= e->size) e->seq = g_nwlog; g_ci_hit[caller]++; return e->off; }   // good up to now: the next lookup scans only later writes
        why = MR_WLOG;
    } else why = !e->gen || (e->addr != addr && e->gen != g_wgen && tog(TG_DC2)) ? MR_EMPTY : e->addr != addr ? MR_OTHER : e->gen != g_wgen || e->seq > g_nwlog ? MR_GEN : e->size < size ? MR_SIZE : MR_ALIGN;
    g_ci_miss[caller]++; g_ci_mbytes[caller] += size; g_ci_why[why]++;
    g_h0_ci[caller & 7][why & 7]++; if (why == MR_GEN) g_h0_mrgen[e->gen >= g_h0_sub_gen0]++;   // H0
    // the pool class (vk19.59c): the entry's when it held this address, else looked up for one miss in 8 and counted for 8 (no lock and pool scan on every miss)
    static uint32_t pc_tick; int cls = e->addr == addr && e->cls < PC_N ? e->cls : PC_N;
    if (cls < PC_N) g_ci_pool[cls]++; else if (!(pc_tick++ & 7)) { cls = pool_class(addr); g_ci_pool[cls] += 8; }
    g_shw_caller = caller; shadow_note(0, addr, size);
    VkDeviceSize o = ring_put((const void *)(uintptr_t)addr, size, align);
    if (o != VK_WHOLE_SIZE) *e = (dcopy){addr, size, g_wgen, o, g_nwlog, (uint8_t)cls};
    return o;
}
static uint64_t g_tex_miss[4];   // texture slots read as the black dummy: 0 no handle bound, 1 handle not registered, 2 texture not created, 3 no view (logged with the frame stats)
// Bindless handles a shader reads from a uniform buffer (tex_ubo / samp_ubo): the word in the bound buffer as guest memory holds it now (the command list's in-order writes
// are applied as it is walked, so this is what the GPU would read at this draw). A separate-sampler handle (type 1) is two words, the texture's and the sampler's, which the
// hardware ORs before taking either ID (as nvn_refl.m pairs them for the Metal host): FINAL FANTASY XII THE ZODIAC AGE's scene composite keeps the depth texture's ID in the
// sampler word. IDs are numbered in the descriptor pools bound when the draw runs (OP_SET_TEX_POOL / OP_SET_SAMP_POOL), as nvn_mtl.m ubo_tex_id
static uint32_t ubo_word(int stage, int reg, int off) {
    if (stage < 0 || stage >= 6 || reg < 0 || reg >= 16 || off < 0) return 0;
    const p_bind_buf *u = &S.ubo[stage][reg]; uint32_t v = 0;
    if (u->addr && (uint64_t)off + 4 <= u->size) memcpy(&v, (const void *)(uintptr_t)(u->addr + (uint64_t)off), 4);
    return v;
}
static uint32_t texu_handle(int stage, const sbind *b) {
    uint32_t h = ubo_word(stage, b->reg, b->off);
    if (b->skind == S_SAMP_UBO && b->soff >= 0 && (b->sreg != b->reg || b->soff != b->off)) h |= ubo_word(stage, b->sreg, b->soff);
    return h;
}
// Diagnostics (vk-t10, always on, a few lines): a single handle a shader reads from a uniform buffer that holds no texture ID when the draw is recorded (the stand-in is bound).
// Tomodachi Life's island composite (adfa17e8e8976a0f bindings 12-15, f11413f932b50da7 2-3: cbuf 0 + 40928..40952) reads 0 there on Android, while the Metal host found
// textures (IDs 1883, 1884) and drew the beach. Each such word (up to 16) is logged once with what was written over it in this submission (inline updates / copies, wlog), and
// read again at the next presents: "written later" means the game fills it after the submission is recorded (the GPU would read it at run time)
static struct { uint64_t addr, fr, sub; char key[20]; int bind, st, done; } g_tz[16]; static int g_ntz;
static void texu_zero_note(int stage, const char *key, const sbind *b) {
    if (stage < 0 || stage >= 6 || b->reg < 0 || b->reg >= 16) return;
    const p_bind_buf *u = &S.ubo[stage][b->reg]; if (!u->addr) return;
    uint64_t a = u->addr + (uint64_t)b->off;
    for (int i = 0; i < g_ntz; i++) if (g_tz[i].addr == a) return;
    if (g_ntz >= 16) return;
    int w = 0; for (uint32_t i = 0; i < g_nwlog && i < NWLOG; i++) if (g_wlo[i] < a + 8 && g_whi[i] > a) w++;
    uint32_t v[4] = {0}; if ((uint64_t)b->off + 16 <= u->size) memcpy(v, (const void *)(uintptr_t)a, 16);
    g_tz[g_ntz].addr = a; g_tz[g_ntz].fr = atomic_load(&g_presents); snprintf(g_tz[g_ntz].key, sizeof g_tz[g_ntz].key, "%s", key); g_tz[g_ntz].bind = (int)b->binding; g_tz[g_ntz].st = stage; g_tz[g_ntz].done = 0; g_tz[g_ntz].sub = g_fence + 1; g_ntz++;
    LOG("vk: handle diag: shader %s binding %u (stage %d) reads cbuf %d +%d = %#llx: words %08x %08x %08x %08x at recording; %d in-order writes over it in this submission; buffer %#llx +%llu",
        key, b->binding, stage, b->reg, b->off, (unsigned long long)a, v[0], v[1], v[2], v[3], w, (unsigned long long)u->addr, (unsigned long long)u->size);
}
static void texu_zero_commit(void) {   // mtl_sub_commit: the words noted in this submission, read again as it is submitted (an in-order write the walk missed lands in between)
    for (int i = 0; i < g_ntz; i++) { if (g_tz[i].sub != g_fence + 1) continue; uint32_t v[2]; memcpy(v, (const void *)(uintptr_t)g_tz[i].addr, 8);
        LOG("vk: handle diag: %s binding %d, %#llx, at the submission's commit: %08x %08x%s", g_tz[i].key, g_tz[i].bind, (unsigned long long)g_tz[i].addr, v[0], v[1], v[0] & 0xfffff ? " (a texture ID: written between the draw's recording and the commit)" : ""); }
}
static void texu_zero_recheck(void) {   // mtl_present: the words noted, read again 1 and 30 frames later
    uint64_t fr = atomic_load(&g_presents);
    for (int i = 0; i < g_ntz; i++) { if (g_tz[i].done >= 2) continue; uint64_t age = fr - g_tz[i].fr; if (age < 1 || (g_tz[i].done == 1 && age < 30)) continue;
        uint32_t v[2]; memcpy(v, (const void *)(uintptr_t)g_tz[i].addr, 8);
        LOG("vk: handle diag: %s binding %d, %#llx, %llu frames later: %08x %08x%s", g_tz[i].key, g_tz[i].bind, (unsigned long long)g_tz[i].addr, (unsigned long long)age, v[0], v[1], v[0] & 0xfffff ? " (a texture ID now: written after the draw was recorded)" : "");
        g_tz[i].done++; }
}
static uint64_t texu_slot(int stage, const sbind *b) {   // as a bound slot's handle: texture ID | sampler ID << 20, both in the host table
    uint32_t h = texu_handle(stage, b), t = h & 0xfffff, sm = h >> 20;
    return (uint64_t)(t ? t + (uint32_t)S.tex_base : 0) | (uint64_t)(sm ? sm + (uint32_t)S.samp_base : 0) << 20;
}
// An array of handles in a uniform buffer that the shader indexes at run time (dmsh_build.py tex_ubo with a count; Ryujinx's texture array from a buffer): element k is the
// handle at byte off + 8k (an NVN handle is 64-bit; the IDs are in its low word), its sampler the same word's high bits unless a separate sampler word is named (soff + 8k).
// As nvn_mtl.m fills it for Metal: only an element whose texture ID is registered is looked up -- most elements are other data the shader never picks as a handle -- and the
// rest are the binding's stand-in. Tomodachi Life: Living the Dream's island ground (c7f9b506209fc76d binding 18) and sea (bb62dd4dd4f94aa7, 1b611e4eb2a2d0eb) read 32 each;
// left unwritten they read black
static uint64_t texu_elem_h(int stage, const sbind *b, int k) {   // element k as a bound slot's handle (texture ID | sampler ID << 20 in the host table), not checked: callers holding g_rmu
    uint32_t h = ubo_word(stage, b->reg, b->off + 8 * k);
    if (b->skind == S_SAMP_UBO && b->soff >= 0 && (b->sreg != b->reg || b->soff != b->off)) h |= ubo_word(stage, b->sreg, b->soff + 8 * k);
    uint32_t t = h & 0xfffff, sm = h >> 20;
    if (!t) return 0;
    return (uint64_t)(t + (uint32_t)S.tex_base) | (uint64_t)(sm ? sm + (uint32_t)S.samp_base : 0) << 20;
}
static uint64_t texu_elem(int stage, const sbind *b, int k) {   // the same, 0 unless its texture ID is registered (takes g_rmu)
    uint64_t h = texu_elem_h(stage, b, k); uint32_t t = (uint32_t)(h & 0xfffff); if (!t) return 0;
    lk_(&g_rmu, LW_RMU); int ok = t < (uint32_t)g_ntreg && g_treg[t].ok; pthread_mutex_unlock(&g_rmu);
    return ok ? h : 0;
}
// DM_VK_HTAB=1 (vk19.52, opt-in; ffr-par's stage 0b): texture and sampler handles resolved once a submission. A texture slot was a registration lock with a 240-byte copy, tex_get
// (another lock, the table lookup, the eviction and dirty checks) and tex_view (a walk of the image's views) for every element of every binding, 60-80 a stage. The table keeps,
// per texture ID, the image and view the first lookup of the submission found, valid while no registration changed (g_reg_gen); a hit touches the image as tex_get does (kept
// from eviction for this submission) and goes the full way when the image was evicted, released, written since its upload and not yet checked this submission, or is a
// CPU-written linear texture. Samplers by ID likewise, without the lock. Counted in the descriptor line
// On by default from vk19.67 (session 2: at matched cpu7 clocks, recording -0.9 and RHIThread -1.0 ms); DM_VK_HTAB=0 resolves every handle the full way again
static int htab_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_HTAB"); v = !(e && atoi(e) == 0); if (v) LOG("vk: DM_VK_HTAB on: texture and sampler handles resolved once a submission"); else LOG("vk: DM_VK_HTAB=0: every texture and sampler handle resolved the full way"); } return v; }
static struct { uint32_t tid; uint64_t gen, reg; vtex *v; VkImageView view; int ok; } g_ht[8192];
static struct { int sid; uint64_t gen, reg; VkSampler s; } g_hs[1024];
static uint64_t g_ht_hit, g_ht_miss, g_ht_slow;
static VkSampler samp_cached(int sid) {
    uint64_t reg = atomic_load_explicit(&g_reg_gen, memory_order_relaxed); typeof(g_hs[0]) *e = &g_hs[(unsigned)sid % 1024];
    if (e->sid == sid && e->gen == g_bvgen && e->reg == reg && e->s) return e->s;
    lk_(&g_rmu, LW_RMU); VkSampler sm = samp_get(sid); pthread_mutex_unlock(&g_rmu);
    e->sid = sid; e->gen = g_bvgen; e->reg = reg; e->s = sm; return sm;
}
static VkImageView tex_slot_(int stage, uint64_t h, VkSampler *sm, int *ok, vtex **vo);
static VkImageView tex_slot(int stage, uint64_t h, VkSampler *sm, int *ok, vtex **vo) {
    uint32_t tid = (uint32_t)(h & 0xfffff);
    if (!htab_on() || !tid) return tex_slot_(stage, h, sm, ok, vo);
    uint64_t reg = atomic_load_explicit(&g_reg_gen, memory_order_relaxed); typeof(g_ht[0]) *e = &g_ht[(tid * 2654435761u >> 19) % 8192];
    if (e->tid == tid && e->gen == g_bvgen && e->reg == reg) {
        vtex *v = e->v; uint64_t pres = atomic_load(&g_presents);
        if (!v) { *ok = e->ok; if (vo) *vo = NULL; g_ht_hit++; return e->view; }
        if (!(v->evicted || v->dirty || v->dead || v->late || (v->lcpu && !(v->span && lcpu_dirty_on()) && v->lhash_gen != g_bvgen) || (!v->gpu && v->span && v->clean_gen != g_bvgen && !(v->reup_frame && v->reup_frame == pres && !v->lcpu)))) {
            v->last_frame = pres; v->last_fence = g_fence + 1; *ok = e->ok; if (vo) *vo = v; g_ht_hit++; return e->view; }
        g_ht_slow++;
    } else g_ht_miss++;
    vtex *v = NULL; VkImageView view = tex_slot_(stage, h, sm, ok, &v); if (vo) *vo = v;
    e->tid = tid; e->gen = g_bvgen; e->reg = reg; e->v = v; e->view = view; e->ok = *ok;
    return view;
}
static VkImageView tex_slot_(int stage, uint64_t h, VkSampler *sm, int *ok, vtex **vo) {
    int tid = (int)(h & 0xfffff); *ok = 0; if (vo) *vo = NULL;
    if (!tid) { g_tex_miss[0]++; return g_dummy_view; }
    lk_(&g_rmu, LW_RMU);
    treg_t tr = tid < g_ntreg ? g_treg[tid] : (treg_t){0};
    pthread_mutex_unlock(&g_rmu);
    if (!tr.ok) { g_tex_miss[1]++; return g_dummy_view; }
    vtex *v = tex_get(&tr.t, 0); if (!v) { g_tex_miss[2]++; return g_dummy_view; }
    if (vo) *vo = v;
    VkImageView view = tex_view(v, &tr.t, &tr.v, tr.has_v, -1, 0);
    (void)sm; (void)stage; *ok = view != VK_NULL_HANDLE;
    if (!view) g_tex_miss[3]++;
    return view ? view : g_dummy_view;
}
// Same-pass feedback: a draw that samples a colour attachment of its own pass. On the Switch it reads what was there before the draw; Vulkan leaves it undefined, and the Mali-G57
// returned garbage (Super Mario Odyssey's final pass, agl's post-effect program sampling the image it writes, drew yellow, green and red blotches over the picture; the Metal host
// needs the same for 3D World and Captain Toad, profile key feedback_snap). Here it holds for every game: the pass is split before such a draw, the attachment's level and layer
// copied into a snapshot image (one per format and size, reused), and the draw samples the snapshot. Not done when the draw's colour mask leaves that attachment alone, nor again
// while no draw has written the attachment since its last snapshot. A 3D attachment is left as it was
typedef struct { VkFormat f; uint32_t w, h; VkImage img; VkDeviceMemory m; VkImageView view; vtex *src; uint32_t lv, ly; } fbsnap_t;
static fbsnap_t g_fbs[16]; static int g_nfbs;
static vtex *g_fb_src[8]; static VkImageView g_fb_view[8]; static int g_fb_n;
static uint64_t g_fb_copies;
static fbsnap_t *fb_snap_for(vtex *v, int i) {   // the snapshot image for attachment i of the current pass, taken now unless the one it holds is still current
    uint32_t lv = P.clv[i], ly = P.cly[i], w = v->w >> lv ? v->w >> lv : 1, h = v->h >> lv ? v->h >> lv : 1;
    fbsnap_t *e = NULL;
    for (int k = 0; k < g_nfbs; k++) if (g_fbs[k].src == v && g_fbs[k].lv == lv && g_fbs[k].ly == ly && g_fbs[k].f == v->f.f && g_fbs[k].w == w && g_fbs[k].h == h) { e = &g_fbs[k]; break; }
    if (e && !P.cwr[i]) return e;
    if (!e) for (int k = 0; k < g_nfbs; k++) if (g_fbs[k].f == v->f.f && g_fbs[k].w == w && g_fbs[k].h == h) { e = &g_fbs[k]; break; }   // same shape: reuse its image
    if (!e) {
        if (g_nfbs == 16) { RED_ONCE(40, "same-pass feedback: more than 16 snapshot shapes; the draw samples its attachment directly"); return NULL; }
        e = &g_fbs[g_nfbs]; memset(e, 0, sizeof *e); e->f = v->f.f; e->w = w; e->h = h;
        VkImageCreateInfo ii = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, v->f.f, {w, h, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
                                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
        if (vkCreateImage(g_dev, &ii, NULL, &e->img) != VK_SUCCESS) { RED_ONCE(41, "same-pass feedback: snapshot image not created"); return NULL; }
        VkMemoryRequirements mr; vkGetImageMemoryRequirements(g_dev, e->img, &mr); e->m = mem_alloc(mr, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT); atomic_fetch_add(&g_fbs_bytes, mr.size); VKCHECK(vkBindImageMemory(g_dev, e->img, e->m, 0));
        VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, e->img, VK_IMAGE_VIEW_TYPE_2D, v->f.f, {0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        VKCHECK(vkCreateImageView(g_dev, &vi, NULL, &e->view));
        to_general(C->cb, e->img, VK_IMAGE_ASPECT_COLOR_BIT); g_nfbs++;
        LOG("vk: same-pass feedback: snapshot %ux%u format %d made (serial %llu)", w, h, (int)v->f.f, (unsigned long long)v->serial);
    }
    VkImageCopy c = {{VK_IMAGE_ASPECT_COLOR_BIT, lv, ly, 1}, {0, 0, 0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {w, h, 1}};
    vkCmdCopyImage(C->cb, v->img, VK_IMAGE_LAYOUT_GENERAL, e->img, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
    e->src = v; e->lv = lv; e->ly = ly; P.cwr[i] = 0; g_fb_copies++;
    return e;
}
// Whether a draw sampling texture v through the view tr holds really reads attachment a's level and layer, as nvn_mtl.m fb_prepare decides: 1 = snapshot it (fb_snap_for).
// A view whose levels or layers exclude the attachment's is no feedback. A view over several levels -- a mip-chain pass, level n drawn while level n-1 is sampled, the view
// covering the chain -- is not snapshotted either (a 1-level snapshot in place of the chain read the wrong level), but on Vulkan the sampled view must not include the level
// drawn: a view overlapping the attachment's subresource in the same pass lost the Adreno 825 (VK_ERROR_DEVICE_LOST in Tomodachi Life's t5, where Metal draws it fine). So the
// draw samples a view of the same image cut to the levels below the one drawn (same base, so the shader's level numbers hold; reading the level drawn or above clamps to the
// last one below, which a downsample chain never does), registered like a snapshot (g_fb_src / g_fb_view). Drawing the view's base level itself, or a layer of a single-level
// multi-layer view, has no such cut: snapshotted as before. Tomodachi Life: Living the Dream builds 64x64 → 1x1, 122x82 → 1x1 and a 66-layer cube chain every frame: a pass
// split and a snapshot per level made 287 render passes a frame on the Adreno (182 on the Mac) and vkCmdBeginRendering ~28 ms a frame
static uint64_t g_fb_chain, g_fb_asis;   // mip-chain draws given a view cut below the level they draw; layered / base-level ones left as they are (counted beside the snapshots)
static int fb_view_hits(const treg_t *tr, vtex *v, int a) {
    uint32_t bl = tr->has_v && tr->v.set_levels ? tr->v.base_level : 0, nl = tr->has_v && tr->v.set_levels ? tr->v.nlevels : (v->levels > bl ? v->levels - bl : 1);
    uint32_t b0 = tr->has_v && tr->v.set_layers ? tr->v.min_layer : 0, nly = tr->has_v && tr->v.set_layers ? tr->v.nlayers : (v->layers > b0 ? v->layers - b0 : 1);
    if (!nl) nl = 1; if (!nly) nly = 1;
    uint32_t L = P.clv[a];
    if (L < bl || L >= bl + nl || P.cly[a] < b0 || P.cly[a] >= b0 + nly) return 0;   // another level or layer of the same texture
    int tgt = tr->has_v && tr->v.set_target ? tr->v.target : tr->t.d.target;
    int layered = tgt == NVN_TEXTURE_TARGET_CUBEMAP || tgt == NVN_TEXTURE_TARGET_CUBEMAP_ARRAY || tgt == NVN_TEXTURE_TARGET_2D_ARRAY || tgt == NVN_TEXTURE_TARGET_1D_ARRAY || tgt == NVN_TEXTURE_TARGET_2D_MULTISAMPLE_ARRAY;
    if (nl == 1 && nly == 1 && !layered) return 1;   // the very level and layer drawn, through a plain 2D view: a snapshot (fb_snap_for's is a 2D image)
    if (L == bl || nl == 1) {   // a cube / array view, or the chain's base level: no cut keeps the level drawn out, and a 2D snapshot in a layered binding is a type the shader
        g_fb_asis++;            // does not declare (undefined; an Adreno may fault): left as it was before the cut (a8c227be), counted
        static int once; if (!once++) LOG("vk: same-pass feedback: a view over levels %u+%u, layers %u+%u (target %d) sampled while level %u layer %u is drawn: left as is (no cut possible, no 2D snapshot in a layered binding)", bl, nl, b0, nly, tgt, L, P.cly[a]);
        return 0;
    }
    for (int q = 0; q < g_fb_n; q++) if (g_fb_src[q] == v) return 0;   // cut already (DM_VK_FB_FAST=2 asks twice)
    view_t vv = tr->has_v ? tr->v : (view_t){0}; vv.set_levels = 1; vv.base_level = (uint8_t)bl; vv.nlevels = (uint8_t)(L - bl);
    VkImageView cut = tex_view(v, &tr->t, &vv, 1, -1, 0);
    if (!cut || g_fb_n >= 8) return 1;   // no view to cut to: a snapshot after all
    g_fb_src[g_fb_n] = v; g_fb_view[g_fb_n++] = cut; g_fb_chain++;
    static int once; if (!once++) LOG("vk: same-pass feedback: a view over levels %u+%u, layers %u+%u sampled while level %u layer %u is drawn: a mip-chain pass, sampled through levels %u+%u (no overlap with the level drawn)", bl, nl, b0, nly, L, P.cly[a], bl, L - bl);
    return 0;
}
static int fb_want_full(shader_t *fs, int want[8]) {   // the attachments the draw samples: each texture binding through tex_get, compared with the pass's attachments
    int any = 0;
    for (int i = 0; i < fs->nb; i++) {
        const sbind *b = &fs->b[i];
        // a bound slot, or a handle read from a uniform buffer: FINAL FANTASY XII THE ZODIAC AGE's full-screen passes sample their own attachment through such handles
        if (b->type != BT_CIS || !((b->kind == K_TEX && b->reg >= 0 && b->reg < NTEX) || b->kind == K_TEXU)) continue;
        int ne = b->kind == K_TEXU && b->count > 1 ? (int)b->count : 1;   // an array of handles in a uniform buffer: every element (texu_elem)
        for (int k = 0; k < ne; k++) {
            int tid = (int)((b->kind == K_TEXU ? (ne > 1 ? texu_elem_h(1, b, k) : texu_slot(1, b)) : S.tex[1][b->reg]) & 0xfffff); if (!tid) continue;
            lk_(&g_rmu, LW_RMU); treg_t tr = tid < g_ntreg ? g_treg[tid] : (treg_t){0}; pthread_mutex_unlock(&g_rmu);
            if (!tr.ok) continue;
            vtex *v = tex_get(&tr.t, 0); if (!v) continue;
            for (int a = 0; a < P.n; a++) if (P.ct[a] == v && P.cv[a] && v->d <= 1 && (S.chmask.mask >> (4 * a) & 0xf) && fb_view_hits(&tr, v, a)) { want[a] = 1; any = 1; }
        }
    }
    return any;
}
// DM_VK_FB_FAST (vk19.61, plan v4 C2, off by default; on the run-time list): the same answer without a registration copy and a tex_get per texture binding of every draw.
// The bindings' texture serials are read under one lock and compared with the pass's attachments' serials (two IDs can name one texture: the serial is the texture's); only
// a binding whose serial matches goes the full way (tex_get, the pointer compare as before), so the answer is the full way's. A draw that samples none is remembered per
// fragment shader with what decides it -- its stage's bindings (g_bgen[1]), the registrations (g_reg_gen), the texture pool base, the pass's attachments, views and colour
// mask -- and the next draw with all of them the same skips even the serials. Shaders reading texture handles from a uniform buffer (K_TEXU) are not remembered: their handles
// change with the buffer's bytes. =2 (verify): both ways, the full way's answer used, differences counted
static uint64_t g_fbf_draws, g_fbf_cached, g_fbf_serial, g_fbf_full, g_fbf_chk, g_fbf_bad;
static struct { shader_t *fs; uint64_t bgen, reg; int tb, n; vtex *ct[8]; VkImageView cv[8]; uint64_t ser[8], mask; } g_fbc[256];   // ser: the attachments' serials beside their pointers (a freed vtex's address reused: vk19.61b)
static int fb_want_fast(shader_t *fs, int want[8]) {
    uint64_t reg = atomic_load_explicit(&g_reg_gen, memory_order_relaxed), mask = S.chmask.mask, aser[8];
    for (int a = 0; a < 8; a++) aser[a] = a < P.n && P.ct[a] ? P.ct[a]->serial : 0;
    typeof(g_fbc[0]) *c = &g_fbc[((uintptr_t)fs >> 4 ^ (uintptr_t)fs >> 12) % 256];
    if (c->fs == fs && c->bgen == g_bgen[1] && c->reg == reg && c->tb == S.tex_base && c->n == P.n && c->mask == mask &&
        !memcmp(c->ct, P.ct, sizeof c->ct) && !memcmp(c->cv, P.cv, sizeof c->cv) && !memcmp(c->ser, aser, sizeof aser)) { g_fbf_cached++; return 0; }
    uint64_t ser[8]; int na = 0, cand[64], nc = 0, texu = 0;
    for (int a = 0; a < P.n; a++) ser[a] = P.ct[a] && P.cv[a] && P.ct[a]->d <= 1 && (mask >> (4 * a) & 0xf) ? P.ct[a]->serial : 0, na += ser[a] != 0;
    if (na) {
        lk_(&g_rmu, LW_RMU);
        for (int i = 0; i < fs->nb; i++) {
            const sbind *b = &fs->b[i];
            if (b->type != BT_CIS || !((b->kind == K_TEX && b->reg >= 0 && b->reg < NTEX) || b->kind == K_TEXU)) continue;
            if (b->kind == K_TEXU) texu = 1;
            int ne = b->kind == K_TEXU && b->count > 1 ? (int)b->count : 1;   // an array of handles: every element (texu_elem_h; g_rmu is held)
            for (int k = 0; k < ne; k++) {
                int tid = (int)((b->kind == K_TEXU ? (ne > 1 ? texu_elem_h(1, b, k) : texu_slot(1, b)) : S.tex[1][b->reg]) & 0xfffff); if (!tid || tid >= g_ntreg || !g_treg[tid].ok) continue;
                uint64_t s = g_treg[tid].t.serial; for (int a = 0; a < P.n; a++) if (ser[a] && ser[a] == s) { if (nc < 64) cand[nc++] = tid; break; }
            }
        }
        pthread_mutex_unlock(&g_rmu);
    } else for (int i = 0; i < fs->nb; i++) if (fs->b[i].type == BT_CIS && fs->b[i].kind == K_TEXU) texu = 1;
    int any = 0;
    for (int k = 0; k < nc; k++) {   // the full way for these few
        lk_(&g_rmu, LW_RMU); treg_t tr = cand[k] < g_ntreg ? g_treg[cand[k]] : (treg_t){0}; pthread_mutex_unlock(&g_rmu);
        if (!tr.ok) continue;
        vtex *v = tex_get(&tr.t, 0); if (!v) continue;
        for (int a = 0; a < P.n; a++) if (P.ct[a] == v && P.cv[a] && v->d <= 1 && (mask >> (4 * a) & 0xf) && fb_view_hits(&tr, v, a)) { want[a] = 1; any = 1; }
    }
    if (nc) g_fbf_full++; else g_fbf_serial++;
    // remembered only when no binding's serial matched (vk19.61b): a candidate whose tex_get failed or found another vtex (a remake failing under memory pressure, a released
    // texture's vtex awaiting garbage_free beside its new one) answered "none" for a reason the key does not hold
    if (!nc && !texu) { c->fs = fs; c->bgen = g_bgen[1]; c->reg = reg; c->tb = S.tex_base; c->n = P.n; c->mask = mask; memcpy(c->ct, P.ct, sizeof c->ct); memcpy(c->cv, P.cv, sizeof c->cv); memcpy(c->ser, aser, sizeof aser); }
    return any;
}
static void fb_prepare(shader_t *fs) {   // before a draw (pass open): find the sampled colour attachments it writes, snapshot them, and have write_set bind the snapshots
    g_fb_n = 0;
    int want[8] = {0}, any, m = tog(TG_FB_FAST);
    if (!m) any = fb_want_full(fs, want);
    else { g_fbf_draws++; any = fb_want_fast(fs, want);
        if (m == 2) { int w2[8] = {0}, a2 = fb_want_full(fs, w2); g_fbf_chk++;
            if (a2 != any || memcmp(w2, want, sizeof w2)) { g_fbf_bad++; static int nl; if (nl++ < 8) LOG("[RED] vk: DM_VK_FB_FAST=2: shader %s: fast %d, full %d", fs->key, any, a2); }
            any = a2; memcpy(want, w2, sizeof w2); } }
    if (!any) return;
    int need = 0; for (int a = 0; a < P.n; a++) if (want[a] && P.cwr[a]) need = 1;
    if (need) pass_end();
    VkCommandBuffer cb = C->cb;
    for (int a = 0; a < P.n && g_fb_n < 8; a++) if (want[a]) { fbsnap_t *e = fb_snap_for(P.ct[a], a); if (e) { g_fb_src[g_fb_n] = P.ct[a]; g_fb_view[g_fb_n++] = e->view; } }
    if (need) { barrier_xfer(cb); pass_begin(); }   // the snapshot copies (fb_snap_for), for the draw's sampling
}
static gw_t g_wb[16]; static int g_nwb;   // the storage buffers the last compute descriptor set bound (write_set), which its dispatch writes
// Texel buffers (NVN buffer textures: Unreal's light lists, particle and skinning data, its GPU scene): the texture's elements copied into the ring -- at the texel-buffer offset
// alignment, which a game's own pool offsets need not keep (the Metal host shadows them for the same reason) -- and a buffer view made over the copy for this submission. A
// compute shader's storage texel buffer is written back like its storage buffers (g_wb); a draw's writes are lost. Views are cached per submission by buffer, offset, format and size
static struct { VkBuffer b; VkDeviceSize off, n; VkFormat f; uint64_t gen; VkBufferView v; } g_bvc[1024];
static VkBufferView bview(VkBuffer b, VkDeviceSize off, VkDeviceSize n, VkFormat f) {
    unsigned h = (unsigned)(((uintptr_t)b >> 4) ^ (off >> 4) ^ (off >> 14) ^ (unsigned)f * 2654435761u) % 1024;
    if (g_bvc[h].gen == g_bvgen && g_bvc[h].b == b && g_bvc[h].off == off && g_bvc[h].n == n && g_bvc[h].f == f) return g_bvc[h].v;
    VkBufferViewCreateInfo ci = {VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO, NULL, 0, b, f, off, n};
    atomic_fetch_add_explicit(&g_bv_made, 1, memory_order_relaxed);
    VkBufferView v; if (vkCreateBufferView(g_dev, &ci, NULL, &v) != VK_SUCCESS) { RED_ONCE(70, "texel buffer view (format %d, %llu bytes) not created", (int)f, (unsigned long long)n); return VK_NULL_HANDLE; }
    if (C->nbv == C->cbv) { C->cbv = C->cbv ? 2 * C->cbv : 256; C->bv = realloc(C->bv, C->cbv * sizeof *C->bv); }
    C->bv[C->nbv++] = v; g_bvc[h] = (typeof(g_bvc[0])){b, off, n, f, g_bvgen, v};
    return v;
}
static VkFormatProperties fmt_props(VkFormat f) {   // vkGetPhysicalDeviceFormatProperties, cached (asked for every texel buffer of every draw)
    static struct { VkFormat f; int ok; VkFormatProperties p; } c[512];
    unsigned h = (unsigned)f * 2654435761u % 512;
    for (int k = 0; k < 512; k++, h = (h + 1) % 512) {
        if (c[h].ok && c[h].f == f) return c[h].p;
        if (!c[h].ok) { c[h].f = f; vkGetPhysicalDeviceFormatProperties(g_pd, f, &c[h].p); c[h].ok = 1; return c[h].p; }
    }
    VkFormatProperties p; vkGetPhysicalDeviceFormatProperties(g_pd, f, &p); return p;
}
static uint64_t g_tbuf_miss[3]; static uint32_t g_ws_zero_tb;   // g_ws_zero_tb: texel buffer elements the last write_sets bound to the zero stand-in (the trace)   // texel buffer slots read as zeros: 0 nothing bound, 1 not a buffer texture / no format, 2 format not usable as a texel buffer here
// ---------- Texel store (DM_VK_TEXEL_STORE, vk19.70, plan v4 C4'b stage 1; off by default, read at start) ----------
// Uniform texel buffers are copied into the ring at every first use in a submission, with a new VkBufferView each (session 1: 745 texel copies, 5.8 MB and 784 views a frame;
// session 2: 2.50 MB of those bytes repeat from an earlier frame). With the store on, a range whose bytes stayed the same over two consecutive frames is admitted into a
// persistent host-visible buffer and served from there, with views that live as long as its block, while nvn_dirty says no page of it was written since.
// - Admission: a range's first miss records it; its second (in another frame) hashes its page-aligned span (pages it shares with per-frame data change too, and protecting them
//   would fault every frame); the same hash in the next frame admits it. Only uniform texel buffers (no storage texel buffer, nothing gw_find / gw_any knows as GPU-written),
//   tracked pools that are not GPU-only, not with DM_VK_ZEROCOPY.
// - Validity, in this order: stamp (nvn_dirty_now), protect (nvn_dirty_protect, which reports an mprotect failure: never admitted then), read into the block. A use needs the
//   span's registration id unchanged (a pool slot reused by another pool has a new one; checked before since()) and no page written since the stamp. A range found written is
//   copied again (in place, or into a new block while a submission in flight still reads the old one); written three times it is dropped and refused for 10 s.
// - Blocks: 4 KB granules (offsets meet minTexelBufferOffsetAlignment), a direct-mapped table of 4,096 entries, least recently used evicted when full. A block and its views
//   are freed only when every submission that read it is done (its fence, set at each use). The views are the store's own: not in C->bv nor in bview's cache.
// - Pool deletion (mtl_pool_del, any thread) queues the pool's range under g_pool_mu; the recording thread drops its blocks at its next lookup. Memory pressure (g_mem_low /
//   g_mem_tight) drops the whole store: its pages are unprotected (dm_guest_write_prepare), its blocks freed after their fences, the buffer last.
// - =2 (verify): a host-side copy per block; every use compares guest memory with it, counts and logs a difference and takes the copy path for that use.
// Everything runs on the recording thread (tbuf_view); faults only touch nvn_dirty's atomics. Kernel writes into guest memory (EFAULT on a protected page instead of a fault):
// hle_fs ReadFile / ReadFileExact / ReadFileOpt (pread) and hle_cheats' vm_write prepare first; no host code recv()s or ioctl()s into guest memory; arc4random_buf copies in user space
static int gw_any(uint64_t a, uint64_t n);
static VkBuffer g_ts_buf; static VkDeviceMemory g_ts_mem; static uint8_t *g_ts_p; static uint32_t g_ts_ngran, g_ts_next; static uint8_t *g_ts_used; static int g_ts_dropping;
enum { TS_GRAN = 4096, TS_NENT = 4096, TS_NCAND = 8192 };
typedef struct { VkFormat f; VkDeviceSize n; VkBufferView v; } ts_view_t;
typedef struct { uint64_t addr, len, regid, stamp, fence, tick, serial; const void *span; uint32_t off_g, ng; uint8_t used, dirty, nv, kind; ts_view_t v[4]; uint8_t *shadow; } ts_ent;   // kind: TSK_* (vk19.72)
enum { TSK_TEXEL, TSK_VTX, TSK_IDXPOOL };
static ts_ent g_ts[TS_NENT];
static struct { uint64_t addr, len, h, hframe, bl_until; uint32_t nmis; uint8_t kind, nbl; } g_tc[TS_NCAND];   // nbl: times refused (vk19.74a: 10 s, then 60 s, then for the session: vk19.75a, in g_tsbl)   // nmis: hashes that differed from the last (vk19.70b: 3 and it is refused for 10 s)
typedef struct { VkBufferView v[4]; int nv; uint32_t off_g, ng; uint64_t fence; uint8_t kind; } ts_ret_t;
static uint32_t g_ts_bgran;   // granules held by vertex and index-pool blocks: at most half the store (vk19.74b: they never push texel blocks out)
static ts_ret_t *g_ts_ret; static int g_ts_nret, g_ts_cret;   // blocks and views waiting for their readers: grows as needed (vk19.70b: a block is never forgotten)
static ts_ret_t *ts_ret_push(void) {
    if (g_ts_nret == g_ts_cret) { int c = g_ts_cret ? 2 * g_ts_cret : 256; ts_ret_t *n = realloc(g_ts_ret, (size_t)c * sizeof *n); if (!n) die("texel store: retired-block list allocation failed"); g_ts_ret = n; g_ts_cret = c; }
    return &g_ts_ret[g_ts_nret++];
}
static uint64_t g_ts_tick, g_ts_serial, g_ts_hit, g_ts_hit_b, g_ts_miss, g_ts_adm, g_ts_recopy, g_ts_evict, g_ts_bl, g_ts_vmade, g_ts_vreuse, g_ts_vchk, g_ts_vbad, g_ts_refused, g_ts_drops;
static uint32_t g_ts_inuse;   // granules in use
static struct { uint64_t mem, size; } g_ts_pdel[64]; static _Atomic int g_ts_npdel;   // deleted pools to drop (under g_pool_mu)
static int ts_mode(void) { return tog(TG_TSTORE); }
static int ts_kmode(int kind) { return kind == TSK_TEXEL ? tog(TG_TSTORE) : tog(TG_BSTORE); }   // the switch of a kind of range (vk19.72)
static int ts_any(void) { return tog(TG_TSTORE) || tog(TG_BSTORE); }
static uint64_t g_ts_khit[3], g_ts_khit_b[3], g_ts_kadm[3];   // per kind (the store line)
static uint64_t g_ts_sec, g_ts_frame;   // the clock (s) and the present count, taken once a submission (ts_tick, vk19.74c)
static uint64_t ts_now_s(void) { return g_ts_sec; }
static int ts_init(void) {
    if (g_ts_buf) return 1;
    if (zc_mode()) { RED_ONCE(110, "DM_VK_TEXEL_STORE is refused together with DM_VK_ZEROCOPY"); return 0; }
    const char *e = getenv("DM_VK_TEXEL_STORE_MB"); int mb = e && atoi(e) > 0 ? atoi(e) : 32; if (mb > 256) mb = 256;
    g_ts_ngran = (uint32_t)((uint64_t)mb << 20) / TS_GRAN; g_ts_used = calloc(g_ts_ngran, 1); if (!g_ts_used) return 0;
    g_ts_buf = buf_new((VkDeviceSize)g_ts_ngran * TS_GRAN, ALL_BUF_USAGE, &g_ts_mem, (void **)&g_ts_p); g_ts_next = 0; g_ts_inuse = 0; g_ts_bgran = 0; g_ts_dropping = 0;
    LOG("vk: the range store: %d MB for ranges that stay the same across frames (DM_VK_TEXEL_STORE=%d: texel buffers; DM_VK_BUF_STORE=%d: vertex streams and the indirect index pool; 2: each use compared with a host copy)", mb, tog(TG_TSTORE), tog(TG_BSTORE));
    return 1;
}
static void ts_gran_set(uint32_t o, uint32_t n, uint8_t v) { memset(g_ts_used + o, v, n); if (v) g_ts_inuse += n; else g_ts_inuse -= n; }
static int ts_gran_alloc(uint32_t n) {   // next fit over the granule map: -1 when no run of n is free
    if (!n || n > g_ts_ngran) return -1;
    for (uint32_t pass = 0, o = g_ts_next; pass < 2; pass++, o = 0) {
        uint32_t run = 0;
        for (uint32_t i = o; i < g_ts_ngran; i++) { if (g_ts_used[i]) { run = 0; continue; } if (++run == n) { uint32_t s0 = i + 1 - n; ts_gran_set(s0, n, 1); g_ts_next = i + 1 < g_ts_ngran ? i + 1 : 0; return (int)s0; } }
    }
    return -1;
}
static void ts_retire(ts_ent *e) {   // the block and its views go when its readers are done
    if (!e->used) return;
    { ts_ret_t *r = ts_ret_push(); r->nv = e->nv; for (int i = 0; i < e->nv; i++) r->v[i] = e->v[i].v; r->off_g = e->off_g; r->ng = e->ng; r->fence = e->fence; r->kind = e->kind; }
    free(e->shadow); memset(e, 0, sizeof *e);
}
static void ts_reap(void) {   // blocks whose readers are done: views destroyed, granules freed; a dropped store's buffer last
    uint64_t done = __atomic_load_n(&g_done, __ATOMIC_ACQUIRE);
    for (int i = 0; i < g_ts_nret;) { ts_ret_t *r = &g_ts_ret[i];
        if (r->fence > done) { i++; continue; }
        for (int k = 0; k < r->nv; k++) vkDestroyBufferView(g_dev, r->v[k], NULL);
        ts_gran_set(r->off_g, r->ng, 0); if (r->kind != TSK_TEXEL) g_ts_bgran -= r->ng; g_ts_ret[i] = g_ts_ret[--g_ts_nret]; }
    if (g_ts_dropping && !g_ts_nret && g_ts_buf) { vkDestroyBuffer(g_dev, g_ts_buf, NULL); vkFreeMemory(g_dev, g_ts_mem, NULL); g_ts_buf = VK_NULL_HANDLE; g_ts_p = NULL; free(g_ts_used); g_ts_used = NULL; g_ts_dropping = 0;
                                                    LOG("vk: DM_VK_TEXEL_STORE: the store was dropped (memory pressure); made again when it has gone"); }
}
static void ts_drop_all(void) {
    for (int i = 0; i < TS_NENT; i++) if (g_ts[i].used) { dm_guest_write_prepare((void *)(uintptr_t)g_ts[i].addr, (size_t)g_ts[i].len); ts_retire(&g_ts[i]); }   // its pages writable again
    g_ts_dropping = 1; g_ts_drops++;
}
static uint64_t ts_span_hash(uint64_t addr, uint64_t len) {   // the page-aligned span's bytes
    static uint64_t pg; if (!pg) { long v = sysconf(_SC_PAGESIZE); pg = v > 0 ? (uint64_t)v : 4096; }   // the protection's page
    uint64_t a = addr & ~(pg - 1), b = (addr + len + pg - 1) & ~(pg - 1);
    return shadow_hash((const uint8_t *)(uintptr_t)a, b - a);
}
static ts_ent *ts_slot(uint64_t addr, uint64_t len, int kind) { return &g_ts[(addr >> 6 ^ addr >> 19 ^ len * 0x9e3779b1u ^ (uint64_t)kind * 0x5bd1e995u) % TS_NENT]; }
static VkBufferView ts_view(ts_ent *e, VkFormat f, VkDeviceSize n) {
    for (int i = 0; i < e->nv; i++) if (e->v[i].f == f && e->v[i].n == n) { g_ts_vreuse++; return e->v[i].v; }
    VkBufferViewCreateInfo ci = {VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO, NULL, 0, g_ts_buf, f, (VkDeviceSize)e->off_g * TS_GRAN, n};
    VkBufferView v; if (vkCreateBufferView(g_dev, &ci, NULL, &v) != VK_SUCCESS) return VK_NULL_HANDLE;
    g_ts_vmade++;
    if (e->nv == 4) { ts_ret_t *r = ts_ret_push(); r->nv = 1; r->v[0] = e->v[0].v; r->off_g = 0; r->ng = 0; r->fence = e->fence; r->kind = e->kind; memmove(e->v, e->v + 1, 3 * sizeof *e->v); e->nv = 3; }
    e->v[e->nv++] = (ts_view_t){f, n, v};
    return v;
}
static int ts_fill(ts_ent *e, uint64_t addr, uint64_t len, const void *span, uint64_t regid, int fresh, int kind) {   // stamp -> protect -> read, into e's block (or a new one)
    uint32_t ng = (uint32_t)((len + TS_GRAN - 1) / TS_GRAN);
    int newblock = fresh || e->fence > __atomic_load_n(&g_done, __ATOMIC_ACQUIRE) || e->ng < ng;
    uint32_t off_g = e->off_g;
    if (newblock) {
        int bk = kind != TSK_TEXEL;   // a vertex or index-pool block: within half the store, and it pushes out only blocks of its own kinds (vk19.74b)
        int over = bk && g_ts_bgran + ng > g_ts_ngran / 2;
        int o = over ? -1 : ts_gran_alloc(ng);
        for (int tries = 0; o < 0 && tries < 64; tries++) {   // full (or the buffer kinds' half full): the least recently used block whose readers are done
            ts_ent *v = NULL; uint64_t done = __atomic_load_n(&g_done, __ATOMIC_ACQUIRE);
            for (int i = 0; i < TS_NENT; i++) if (g_ts[i].used && &g_ts[i] != e && g_ts[i].fence <= done && (!bk || g_ts[i].kind != TSK_TEXEL) && (!v || g_ts[i].tick < v->tick)) v = &g_ts[i];
            if (!v) break;
            ts_retire(v); g_ts_evict++; ts_reap();
            over = bk && g_ts_bgran + ng > g_ts_ngran / 2; o = over ? -1 : ts_gran_alloc(ng);
        }
        if (o < 0) return 0;
        if (bk) g_ts_bgran += ng;
        if (!fresh) ts_retire(e);   // the old block stays for its readers
        off_g = (uint32_t)o;
    }
    uint64_t stamp = nvn_dirty_now();
    if (!nvn_dirty_protect((const void *)(uintptr_t)addr, (size_t)len)) { if (newblock) { ts_gran_set(off_g, ng, 0); if (kind != TSK_TEXEL) g_ts_bgran -= ng; } if (!newblock || !fresh) ts_retire(e); return 0; }
    memcpy(g_ts_p + (size_t)off_g * TS_GRAN, (const void *)(uintptr_t)addr, (size_t)len);
    if (newblock) { memset(e, 0, sizeof *e); e->off_g = off_g; e->ng = ng; e->serial = ++g_ts_serial; }
    e->addr = addr; e->len = len; e->regid = regid; e->span = span; e->stamp = stamp; e->used = 1; e->tick = ++g_ts_tick; e->fence = g_fence + 1; e->kind = (uint8_t)kind;
    if (ts_kmode(kind) == 2) { if (!e->shadow) e->shadow = malloc((size_t)len); if (e->shadow) memcpy(e->shadow, (const void *)(uintptr_t)addr, (size_t)len); }
    return 1;
}
static uint32_t ts_cand(uint64_t addr, uint64_t len, int kind) {   // vertex and index-pool candidates use the upper half of the table only (vk19.74b)
    uint64_t h = addr >> 6 ^ addr >> 19 ^ len * 0x9e3779b1u ^ (uint64_t)kind * 0x5bd1e995u;
    return kind == TSK_TEXEL ? (uint32_t)(h % TS_NCAND) : (uint32_t)(TS_NCAND / 2 + h % (TS_NCAND / 2)); }
// Ranges refused for the session (vk19.75a): their own small set (128 sets of 4, the oldest of a set replaced), not the candidate table, whose slot a session-long refusal
// used to hold for good (every other range landing there was refused too: over hours of play the table died slot by slot)
static struct { uint64_t addr, len; uint8_t kind, used; } g_tsbl[128][4]; static uint8_t g_tsbl_next[128]; static uint64_t g_tsbl_n;
static uint32_t ts_bl_set(uint64_t addr, uint64_t len, int kind) { return (uint32_t)((addr >> 6 ^ addr >> 21 ^ len * 0x9e3779b1u ^ (uint64_t)kind * 0x5bd1e995u) % 128); }
static int ts_bl_session(uint64_t addr, uint64_t len, int kind) {
    uint32_t st = ts_bl_set(addr, len, kind);
    for (int w = 0; w < 4; w++) if (g_tsbl[st][w].used && g_tsbl[st][w].addr == addr && g_tsbl[st][w].len == len && g_tsbl[st][w].kind == kind) return 1;
    return 0;
}
static void ts_blacklist(uint64_t addr, uint64_t len, int kind) {   // refused for 10 s, again 60 s, a third time for the session (vk19.74a: a range that keeps failing stops costing hashes and copies)
    typeof(g_tc[0]) *c = &g_tc[ts_cand(addr, len, kind)];
    uint8_t nbl = c->addr == addr && c->len == len && c->kind == kind && c->nbl < 255 ? c->nbl + 1 : 1;
    g_ts_bl++;
    if (nbl >= 3) {   // for the session: into the set, and the candidate slot is free for any other range
        uint32_t st = ts_bl_set(addr, len, kind); int w = g_tsbl_next[st]; g_tsbl_next[st] = (uint8_t)((w + 1) & 3);
        g_tsbl[st][w].addr = addr; g_tsbl[st][w].len = len; g_tsbl[st][w].kind = (uint8_t)kind; g_tsbl[st][w].used = 1; g_tsbl_n++;
        memset(c, 0, sizeof *c); return; }
    c->addr = addr; c->len = len; c->kind = (uint8_t)kind; c->h = 0; c->hframe = 0; c->nmis = 0; c->nbl = nbl; c->bl_until = ts_now_s() + (nbl == 1 ? 10 : 60);
}
void ts_pool_deleted(const void *mem, uint64_t size) {   // mtl_pool_del (any thread), holding g_pool_mu: the recording thread drops its blocks at its next lookup
    int n = atomic_load(&g_ts_npdel); if (n < 64) { g_ts_pdel[n].mem = (uint64_t)(uintptr_t)mem; g_ts_pdel[n].size = size; atomic_store(&g_ts_npdel, n + 1); }
    else { g_ts_pdel[63].mem = 0; g_ts_pdel[63].size = UINT64_MAX; }   // too many at once: every block goes
}
static pthread_mutex_t g_pool_mu;
// The store's upkeep, once a submission at mtl_sub_begin (vk19.74c: it ran in every lookup, ~3,000 a frame with vertex streams): the clock and the present count for the
// lookups, memory pressure (the whole store dropped), its making, the blocks of deleted pools, and the blocks whose readers are done
static void ts_tick(void) {
    if (!ts_any() || zc_mode()) return;
    g_ts_sec = now_ns() / 1000000000ull; g_ts_frame = atomic_load_explicit(&g_presents, memory_order_relaxed);
    if (g_mem_low || g_mem_tight) { if (g_ts_buf && !g_ts_dropping) ts_drop_all(); if (g_ts_buf) ts_reap(); return; }
    if (g_ts_dropping) { ts_reap(); return; }
    if (!g_ts_buf && !ts_init()) return;
    if (atomic_load(&g_ts_npdel)) {   // pools deleted since: their blocks go
        pthread_mutex_lock(&g_pool_mu); int k = atomic_load(&g_ts_npdel);
        for (int j = 0; j < k; j++) for (int i = 0; i < TS_NENT; i++) if (g_ts[i].used && g_ts[i].addr + g_ts[i].len > g_ts_pdel[j].mem && g_ts[i].addr < g_ts_pdel[j].mem + g_ts_pdel[j].size) ts_retire(&g_ts[i]);
        atomic_store(&g_ts_npdel, 0); pthread_mutex_unlock(&g_pool_mu); }
    ts_reap();
}
// The range's block in the store (filled now if just admitted or written since), NULL: the caller copies as before. kind: texel buffer, vertex stream, indirect index pool
static ts_ent *ts_get(uint64_t addr, uint64_t n, VkDeviceSize al, int kind) {
    int mode = ts_kmode(kind); if (!mode || !g_ts_buf || g_ts_dropping || g_mem_low || g_mem_tight) return NULL;   // the store's upkeep runs once a submission (ts_tick)
    if (al > TS_GRAN || !n || n > (64u << 20)) return NULL;   // a texel range as in vk19.70 (vk19.74d: no further cap for texel buffers)
    if (kind != TSK_TEXEL && n > (2u << 20)) return NULL;   // vk19.74a: vertex streams and the index pool at most 2 MB a range (an 8 MB index pool cost 1.5-2.5 ms a hash and as much a copy when written)
    ts_ent *e = ts_slot(addr, n, kind);
    if (e->used && e->addr == addr && e->len == n && e->kind == kind) {
        uint64_t rid = nvn_dirty_span_regid(e->span);
        if (!rid || rid != e->regid) { ts_retire(e); g_ts_miss++; return NULL; }   // the pool's registration changed (checked before since: a reused slot starts with zeroed generations)
        int clean = !nvn_dirty_since_span(e->span, (const void *)(uintptr_t)addr, (size_t)n, e->stamp) && !gw_any(addr, n);
        if (clean && mode == 2 && e->shadow) { g_ts_vchk++;
            if (memcmp(e->shadow, (const void *)(uintptr_t)addr, (size_t)n)) { g_ts_vbad++; clean = 0; static int nl; if (nl++ < 8) LOG("[RED] vk: range store, verify: %s range %#llx (%llu bytes) differs from its stored copy while nvn_dirty says unwritten", kind == TSK_TEXEL ? "texel" : kind == TSK_VTX ? "vertex" : "index-pool", (unsigned long long)addr, (unsigned long long)n); } }
        if (clean) { e->fence = g_fence + 1; e->tick = ++g_ts_tick; g_ts_hit++; g_ts_hit_b += n; g_ts_khit[kind]++; g_ts_khit_b[kind] += n; return e; }
        if (++e->dirty >= 3 || gw_any(addr, n)) { ts_retire(e); ts_blacklist(addr, n, kind); g_ts_miss++; return NULL; }   // written again and again: refused for a while
        uint8_t d = e->dirty;
        if (!ts_fill(e, addr, n, e->span, rid, 0, kind)) { g_ts_miss++; return NULL; }
        e->dirty = d; g_ts_recopy++;
        return e;
    }
    g_ts_miss++;
    typeof(g_tc[0]) *c = &g_tc[ts_cand(addr, n, kind)];
    if (c->addr != addr || c->len != n || c->kind != kind) { if (ts_bl_session(addr, n, kind) || c->bl_until > ts_now_s()) return NULL; /* refused for the session (its own set), or the slot's range refused for a while */ c->addr = addr; c->len = n; c->kind = (uint8_t)kind; c->h = 0; c->hframe = 0; c->bl_until = 0; c->nmis = 0; c->nbl = 0; return NULL; }   // its first miss: noted, not hashed
    if (c->bl_until && c->bl_until > ts_now_s()) return NULL;
    uint64_t fr = g_ts_frame;
    if (c->hframe == fr) return NULL;   // hashed in this frame already
    // the cheap refusals before any hash (vk19.75c): larger than half the store (an allocation that would fail), an untracked span, a GPU-only pool, GPU-written bytes
    if (n > (uint64_t)g_ts_ngran * TS_GRAN / 2) { ts_blacklist(addr, n, kind); return NULL; }
    const void *span = nvn_dirty_span((const void *)(uintptr_t)addr, (size_t)n); uint64_t rid = nvn_dirty_span_regid(span);
    if (!span || !rid || gw_any(addr, n) || pool_class(addr) == PC_GPU) { g_ts_refused++; ts_blacklist(addr, n, kind); return NULL; }
    uint64_t h = ts_span_hash(addr, n);
    if (!(c->hframe && c->hframe + 1 == fr && c->h == h)) {
        if (c->hframe && c->h != h && ++c->nmis >= 3) { ts_blacklist(addr, n, kind); return NULL; }   // never stable: not hashed again for a while (a hash costs about a copy)
        c->h = h; c->hframe = fr; return NULL; }
    // the same bytes in two consecutive frames: admitted (span, registration and exclusions checked above, before the hash)
    if (e->used) { if (e->fence > __atomic_load_n(&g_done, __ATOMIC_ACQUIRE) || (kind != TSK_TEXEL && e->kind == TSK_TEXEL)) { g_ts_refused++; return NULL; } ts_retire(e); g_ts_evict++; }   // a vertex / index-pool range never takes a texel block's slot (vk19.74b)
    if (!ts_fill(e, addr, n, span, rid, 1, kind)) { g_ts_refused++; ts_blacklist(addr, n, kind); return NULL; }
    g_ts_adm++; g_ts_kadm[kind]++; c->addr = 0;
    return e;
}
// A texel buffer: the store's own view over the block (DM_VK_TEXEL_STORE)
static int ts_lookup(uint64_t addr, uint64_t n, VkFormat f, VkDeviceSize al, VkBufferView *out) {
    ts_ent *e = ts_get(addr, n, al, TSK_TEXEL); if (!e) return 0;
    VkBufferView v = ts_view(e, f, n); if (!v) return 0;
    *out = v; return 1;
}
// A vertex stream or the indirect index pool: the store's buffer at the block (DM_VK_BUF_STORE, vk19.72)
static int ts_buf(uint64_t addr, uint64_t n, VkDeviceSize al, int kind, VkBuffer *b, VkDeviceSize *o) {
    ts_ent *e = ts_get(addr, n, al, kind); if (!e) return 0;
    *b = g_ts_buf; *o = (VkDeviceSize)e->off_g * TS_GRAN; return 1;
}
static VkBufferView tbuf_view(int stage, uint64_t h, int storage) {
    int tid = (int)(h & 0xfffff); if (!tid) { g_tbuf_miss[0]++; return VK_NULL_HANDLE; }
    lk_(&g_rmu, LW_RMU); treg_t tr = tid < g_ntreg ? g_treg[tid] : (treg_t){0}; pthread_mutex_unlock(&g_rmu);
    if (!tr.ok || tr.t.d.target != NVN_TEXTURE_TARGET_BUFFER || !tr.t.mem) { g_tbuf_miss[tr.ok ? 1 : 0]++; return VK_NULL_HANDLE; }
    int fmtn = tr.has_v && tr.v.set_format ? tr.v.format : tr.t.d.format, bw = 1, bh = 1, bpb = nvn_fmt_bpb(fmtn, &bw, &bh);
    fmt_t f = fmt_of(fmtn);
    if (!f.f || f.dec || f.depth || fmt_block(f.f) || !bpb || bw != 1 || bh != 1) { g_tbuf_miss[1]++; RED_ONCE(71, "buffer texture format %#x: no texel buffer format", fmtn); return VK_NULL_HANDLE; }
    VkFormatProperties fp = fmt_props(f.f);
    if (!(fp.bufferFeatures & (storage ? VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT : VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT))) {
        g_tbuf_miss[2]++; static int n; if (n++ < 4) LOG("[RED] vk: buffer texture format %#x (Vulkan %d) is not a %s texel buffer format on this GPU: reads zero", fmtn, (int)f.f, storage ? "storage" : "uniform"); return VK_NULL_HANDLE; }
    uint64_t el = tr.t.d.width ? tr.t.d.width : 1; if (el > g_props.limits.maxTexelBufferElements) el = g_props.limits.maxTexelBufferElements;
    uint64_t n = el * (uint64_t)bpb, addr = (uint64_t)(uintptr_t)tr.t.mem;
    VkDeviceSize al = g_props.limits.minTexelBufferOffsetAlignment; if (al < 16) al = 16;
    VkBuffer b = VK_NULL_HANDLE; VkDeviceSize off = 0; uint8_t *p = NULL;
    int ci = g_ci, from_gw = 0;
    g_h0_caller = H0C_TEXEL; if (gw_find(addr, n, al, &b, &off, &p)) { ci = g_gw_ci; from_gw = 1; }   // what a dispatch wrote (this submission's, or an earlier one's still running)
    else { if (!storage && ts_mode()) { VkBufferView sv; if (ts_lookup(addr, n, f.f, al, &sv)) return sv; }   // DM_VK_TEXEL_STORE (vk19.70): the store's block and its own view
           off = copy_in(addr, n, al, CI_TEXEL); if (off != VK_WHOLE_SIZE && off % al) off = ring_put((const void *)(uintptr_t)addr, n, al);   // a cached copy at a smaller alignment
           if (off == VK_WHOLE_SIZE) return VK_NULL_HANDLE; b = C->ring; p = C->ringp + off; }
    if (storage && stage == 5 && g_nwb < 16) g_wb[g_nwb++] = (gw_t){addr, n, b, off, p, (uint64_t)from_gw, ci};   // fence field: 1 = GPU-written source (see dispatch)
    return bview(b, off, n, f.f);
}
// Element k of a dynamic slot array: the texture bound to slot k, combined, else separate (as nvn_mtl.m NVN_RB_TEXD)
static uint64_t dyn_slot(int stage, int k) { return k >= 0 && k < 128 ? (S.tex[stage][k] ? S.tex[stage][k] : S.tex[stage][128 + k]) : 0; }
static VkDescriptorType dtype(int t) {
    static const VkDescriptorType DT[] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
                                          VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_SAMPLER};
    return t >= 0 && t < (int)(sizeof DT / sizeof *DT) ? DT[t] : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
}
// The sampler a texture binding pairs with (a combined image-sampler, or a separate sampler binding)
static VkSampler bind_samp(int stage, const sbind *b, uint64_t th, int el) {
    int r = b->sreg >= 0 ? b->sreg + el : -1;
    int sid = b->skind == S_SAMP && r >= 0 && r < 128 ? (int)(S.samp[stage][r] >> 20) : b->skind == S_SAMP_TEX && b->reg >= 0 && b->reg + el < NTEX ? (int)(S.tex[stage][b->reg + el] >> 20) : b->skind == S_SAMP_ID ? b->sreg :
              b->skind == S_SAMP_UBO ? (b->kind == K_TEXU ? (int)(th >> 20) : (int)(ubo_word(stage, b->sreg, b->soff) >> 20) ? (int)(ubo_word(stage, b->sreg, b->soff) >> 20) + S.samp_base : 0) : 0;
    if (htab_on()) return samp_cached(sid);
    lk_(&g_rmu, LW_RMU); VkSampler sm = samp_get(sid); pthread_mutex_unlock(&g_rmu);
    return sm;
}
// Every element of every binding is written: what is bound where it can be found, a stand-in of the binding's type where not. A descriptor left unwritten is undefined, and a
// shader reading one can take the GPU down (FINAL FANTASY RESONANCE lost the Adreno 825 after about 1,200 frames, its Unreal shaders' texel buffers and sampler arrays unwritten)
// Descriptor sets of this submission by what they hold (layout and every descriptor): a draw whose bindings resolve to the same resources reuses the set instead of allocating and
// writing one again -- Unreal's shaders have 60-80 descriptors a stage (32-entry texel buffer and sampler arrays), and writing every one for every draw cost most of the recording time
static struct { uint64_t key, gen; VkDescriptorSetLayout dsl; VkDescriptorSet ds; uint64_t key2; int nk; uint64_t km[64]; } g_dsc[8192];   // km: the key material (DM_VK_DSET_CACHE2)
static uint64_t g_dsc_hit, g_dsc_miss;
static uint64_t g_t_ws, g_t_draw, g_t_disp, g_t_sub;   // time in write_set, in draws (with their write_set), in dispatches, in vkQueueSubmit (the frame stats line)
static VkDescriptorSet write_set_(shader_t *s, int stage);
// DM_VK_SEG_CACHE (vk19.38, default 1): the 4 KB driver segment is a function of the stage's texture, image, sampler and storage-buffer bindings and of its kind. g_bgen[stage]
// changes with any bind of those (OP_BIND_TEX / SAMP / IMG / SSBO, the state reset); a segment built in this submission from the same generation is taken as it was, without
// building, hashing and comparing its 4 KB again (two or three of them for every draw)
static uint64_t g_segc_hit, g_segc_miss;   // g_bgen: declared near the top (fb_prepare reads it too)
static struct { uint64_t gen, bgen; VkBuffer b; VkDeviceSize off; } g_segc[6][3];
static int seg_cache_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_SEG_CACHE"); v = !e || atoi(e) != 0; } return v; }
// DM_VK_DSET_CACHE2 (vk19.36, default 1). The descriptor-set cache (g_dsc) keys a set by what its bindings resolved to, and two kinds of binding were put into the ring anew
// for every draw: the 4 KB driver segment (K_DRVCB: texture / sampler handle tables) and the shader's constant block (K_C1). Each draw then had a new ring offset, so its set
// never matched an earlier one: a new set, a vkAllocateDescriptorSets and a vkUpdateDescriptorSets (6-7 ms a frame at 400 draws, 25-55% reused). Now those blocks are kept by
// content for the submission (the ring is not reused before it is done): the same bytes give the same buffer and offset, so the set can match. A memo hit compares the whole
// block with a host copy (not the ring, which is uncached here). And a set hit compares its whole key material (up to 64 words; past that a second, independent 64-bit hash),
// where it compared one 64-bit hash before. The per-600 line: block reuse, set hits / misses / rejected on compare, and the time the hits saved (the misses' mean)
static int dset_cache2(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_DSET_CACHE2"); v = !e || atoi(e) != 0; LOG("vk: DM_VK_DSET_CACHE2=%d", v); } return v; }
enum { NMEMO = 256, MEMO_MAX = 4096 };
static struct { uint64_t h, gen; uint32_t len; VkBuffer b; VkDeviceSize off; uint8_t *copy; } g_memo[NMEMO];
static uint64_t g_memo_hit, g_memo_miss, g_dsc_rej, g_dsc_miss_ns, g_dsc_nmiss_t;
static VkDeviceSize ring_put_memo(const void *src, uint32_t len, VkDeviceSize align, VkBuffer *buf) {
    *buf = VK_NULL_HANDLE;
    if (!dset_cache2() || len > MEMO_MAX) { VkDeviceSize o = ring_put(src, len, align); if (o != VK_WHOLE_SIZE) *buf = C->ring; return o; }
    uint64_t h = 0xcbf29ce484222325ull, x; uint32_t i = 0;
    if (tog(TG_LEAN)) {   // DM_VK_LEAN: four independent lanes over 32 bytes a step (the one-lane chain waits a multiply per 8 bytes); a match is still checked byte by byte below
        uint64_t l0 = h, l1 = h ^ 0x9e3779b97f4a7c15ull, l2 = h ^ 0xc2b2ae3d27d4eb4full, l3 = h ^ 0x165667b19e3779f9ull, y[4];
        for (; i + 32 <= len; i += 32) { memcpy(y, (const uint8_t *)src + i, 32); l0 = (l0 ^ y[0]) * 0x100000001b3ull; l1 = (l1 ^ y[1]) * 0x100000001b3ull; l2 = (l2 ^ y[2]) * 0x100000001b3ull; l3 = (l3 ^ y[3]) * 0x100000001b3ull; }
        h = (l0 ^ (l1 << 1 | l1 >> 63) ^ (l2 << 2 | l2 >> 62) ^ (l3 << 3 | l3 >> 61)) * 0xff51afd7ed558ccdull; h ^= h >> 33;
    }
    for (; i + 8 <= len; i += 8) { memcpy(&x, (const uint8_t *)src + i, 8); h = (h ^ x) * 0x100000001b3ull; }
    for (; i < len; i++) h = (h ^ ((const uint8_t *)src)[i]) * 0x100000001b3ull;
    h ^= len;
    typeof(g_memo[0]) *m = &g_memo[(h ^ h >> 29) % NMEMO];
    if (m->gen == g_bvgen && m->h == h && m->len == len && m->off % align == 0 && !memcmp(m->copy, src, len)) { g_memo_hit++; *buf = m->b; return m->off; }
    VkDeviceSize o = ring_put(src, len, align);
    if (o == VK_WHOLE_SIZE) return o;
    *buf = C->ring;
    if (!m->copy) m->copy = malloc(MEMO_MAX);
    if (m->copy) { memcpy(m->copy, src, len); m->h = h; m->gen = g_bvgen; m->len = len; m->b = C->ring; m->off = o; }
    g_memo_miss++;
    return o;
}
static int g_ws_att; static vtex *g_ws_simg[16]; static int g_nws_simg;   // the last compute write_set: whether it reads an attachment of the open pass, the storage images it binds (defer_on)
static int ws_is_att(const vtex *v) { if (v == P.dt) return 1; for (int i = 0; i < P.n; i++) if (P.ct[i] == v) return 1; return 0; }
static int g_vs_haz;   // the vertex shader's write_set met a texture a pass wrote with vertex shaders left out of its end barrier (draw_vs_hazard)
static uint32_t g_dyno[64]; static uint32_t g_ndyno;   // the dynamic offsets of the last write_set, in binding order (for vkCmdBindDescriptorSets)
static VkDescriptorSet write_set(shader_t *s, int stage) { uint64_t t0 = stat_ns(); VkDescriptorSet d = write_set_(s, stage); g_t_ws += stat_ns() - t0; return d; }
// DM_VK_STAGE_REUSE (vk19.41, default 1), after the Metal host's stage blocks (nvn_mtl.m: a stage's bindings reused whole while they have not changed). The descriptor set
// of the stage's last draw is bound again without resolving its bindings when the shader, the stage's texture / sampler / image / storage-buffer bindings (g_bgen), the
// registrations (g_reg_gen), the descriptor pool bases and the submission are the same. Only its dynamic uniform buffers are resolved again (their ring offsets go with the
// bind; a buffer other than the set's means a new set), and its textures are touched as tex_get would: kept from eviction for this submission, the vertex-hazard and deferral
// checks; one evicted, written since its upload, or not yet checked this submission sends the draw down the full path. Sets with storage buffers or images, texel buffers,
// bindless (dyn) slots, same-pass feedback snapshots, compute, or a pass log are never reused this way
// ck: bindings checked again on reuse (vk19.43): a uniform texel buffer element (its view resolved again: the same view, or a new set) and a storage buffer a draw stage only
// reads (its buffer, offset and range resolved again: the same, or a new set)
typedef struct { uint8_t kind; uint16_t i, e; VkBufferView v; VkBuffer b; VkDeviceSize off, range; } wck_t;
typedef struct { const shader_t *s; uint64_t bgen, reg, bvgen; int64_t tb, sb; VkDescriptorSet ds; int ok; int nd; uint16_t di[16]; VkBuffer db[16]; VkDeviceSize dr[16]; int nv; vtex *v[64]; int nck; wck_t ck[16]; } wsc_t;
enum { WSC_NF_SIMG, WSC_NF_STB, WSC_NF_TB, WSC_NF_SSBO, WSC_NF_TEXU, WSC_NF_UBO, WSC_NF_MANY, WSC_NF_N };
static uint64_t g_wsc_nf[WSC_NF_N];
static VkBufferView tbuf_view(int stage, uint64_t h, int storage);
static wsc_t g_wsc[6]; static uint64_t g_wsc_hit, g_wsc_miss, g_wsc_nofit;
static int stage_reuse(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_STAGE_REUSE"); v = !e || atoi(e) != 0; LOG("vk: DM_VK_STAGE_REUSE=%d", v); } return v; }
static int dyn_resolve(shader_t *s, int stage, int i, VkBuffer *buf, VkDeviceSize *off, VkDeviceSize *range);
static VkDescriptorSet wsc_try(shader_t *s, int stage) {
    wsc_t *c = &g_wsc[stage];
    if (!c->ok || c->s != s || c->bgen != g_bgen[stage] || c->bvgen != g_bvgen || c->reg != atomic_load_explicit(&g_reg_gen, memory_order_relaxed) || c->tb != S.tex_base || c->sb != S.samp_base) return VK_NULL_HANDLE;
    uint64_t pres = atomic_load(&g_presents);
    for (int k = 0; k < c->nv; k++) { vtex *v = c->v[k];
        if (v->evicted || v->dirty || v->dead || (!v->gpu && v->span && v->clean_gen != g_bvgen && !(v->reup_frame && v->reup_frame == pres && !v->late && !v->lcpu))) return VK_NULL_HANDLE; }
    uint32_t dy[16];
    for (int k = 0; k < c->nd; k++) { VkBuffer b; VkDeviceSize o, r; if (!dyn_resolve(s, stage, c->di[k], &b, &o, &r) || b != c->db[k] || r != c->dr[k]) return VK_NULL_HANDLE; dy[k] = (uint32_t)o; }
    for (int k = 0; k < c->nck; k++) { const wck_t *q = &c->ck[k]; const sbind *b = &s->b[q->i];
        if (q->kind == 0) { int r = b->reg + q->e; uint64_t h = b->kind == K_TBUF ? S.tex[stage][r] : S.img[stage][r]; if (tbuf_view(stage, h, 0) != q->v) return VK_NULL_HANDLE; }
        else { const p_bind_buf *u = b->kind == K_SBUF ? &S.ssbo[stage][b->reg] : &S.ubo[stage][b->reg]; VkBuffer gb = VK_NULL_HANDLE; VkDeviceSize go = 0;
               VkDeviceSize al = b->type == BT_UBO ? g_props.limits.minUniformBufferOffsetAlignment : g_props.limits.minStorageBufferOffsetAlignment; if (!al) al = 256;
               g_h0_caller = b->type == BT_UBO ? H0C_UBO : H0C_SSBO; if (u->addr && u->size && gw_find(u->addr, u->size, al, &gb, &go, NULL)) {}
               else if (u->addr && u->size) { go = copy_in(u->addr, u->size, 256, b->type == BT_UBO ? CI_UBO : CI_SSBO); if (go == VK_WHOLE_SIZE) return VK_NULL_HANDLE; gb = C->ring; }
               if (gb != q->b || go != q->off || u->size != q->range) return VK_NULL_HANDLE; } }
    for (int k = 0; k < c->nv; k++) { vtex *v = c->v[k]; v->last_frame = pres; v->last_fence = g_fence + 1;
        if (stage == 0 && v->vhaz && v->vhaz_epoch == g_vbar_epoch) g_vs_haz = 1;
        if (g_ndefer && v->dfr_gen == g_defer_gen) g_defer_haz = 1; }
    memcpy(g_dyno, dy, (size_t)c->nd * 4); g_ndyno = (uint32_t)c->nd; g_wsc_hit++;
    return c->ds;
}
static void wsc_keep(wsc_t *nc, int fit, const shader_t *s, int stage, VkDescriptorSet ds) {
    if (!stage_reuse() || stage >= 5) return;
    if (!fit) { g_wsc[stage].ok = 0; if (stage_reuse()) g_wsc_nofit++; return; }
    nc->s = s; nc->bgen = g_bgen[stage]; nc->reg = atomic_load_explicit(&g_reg_gen, memory_order_relaxed); nc->bvgen = g_bvgen; nc->tb = S.tex_base; nc->sb = S.samp_base; nc->ds = ds; nc->ok = 1;
    g_wsc[stage] = *nc;
}
// A dynamic uniform binding resolved as write_set_ resolves it (its buffer, ring offset and range), for a set bound again (wsc_try); 0: not a kind kept so
static int dyn_resolve(shader_t *s, int stage, int i, VkBuffer *buf, VkDeviceSize *off, VkDeviceSize *range) {
    const sbind *b = &s->b[i];
    if (b->kind == K_CBUF && b->reg >= 0 && b->reg < 16) {
        const p_bind_buf *u = &S.ubo[stage][b->reg]; uint64_t n = u->size > 65536 ? 65536 : u->size; VkBuffer gb; VkDeviceSize go;
        g_h0_caller = H0C_UBO; if (u->addr && n && gw_find(u->addr, n, g_props.limits.minUniformBufferOffsetAlignment ? g_props.limits.minUniformBufferOffsetAlignment : 256, &gb, &go, NULL)) { *buf = gb; *off = go; *range = n; return 1; }
        if (u->addr && n) { VkDeviceSize o = copy_in(u->addr, n, 256, CI_UBO); if (o == VK_WHOLE_SIZE) return 0; *buf = C->ring; *off = o; *range = n; return 1; }
        *buf = b->type == BT_SSBO ? g_scratch_buf : g_zero_buf; *off = 0; *range = 65536; return 1;
    }
    if (b->kind == K_C1 && s->c1_len) { VkBuffer rb; VkDeviceSize o = ring_put_memo(s->c1, (uint32_t)s->c1_len, 256, &rb); if (o == VK_WHOLE_SIZE) return 0; *buf = rb; *off = o; *range = s->c1_len; return 1; }
    if (b->kind == K_DRVCB) {
        int skind = b->off == 2 ? (s->dyn ? 0 : 1) : 2;
        if (!seg_cache_on() || stage >= 6 || g_segc[stage][skind].gen != g_bvgen || g_segc[stage][skind].bgen != g_bgen[stage]) return 0;
        *buf = g_segc[stage][skind].b; *off = g_segc[stage][skind].off; *range = 4096; return 1;
    }
    return 0;
}
// The pool sets (see pool_dsl): one per (texture pool base, sampler pool base) the game binds, up to POOL_SETS, each kept for its base -- a draw recorded under one base must not see
// another's IDs (with update-after-bind a submission's draws read a set as it is at submit). Made at the first draw of a pool shader under that base, filled with the classes'
// stand-ins, then kept up to date by pool_set at each such draw
typedef struct {
    VkDescriptorSet ds; VkImageView *view[SPV_PC_N]; vtex **v; VkSampler *samp; int tb, sb, synced, pending; uint64_t reg, kill, cpu, sgen, pres, last_fence, used;
} pset_t;
static VkDescriptorPool g_pool_dp; static VkImageView g_pool_dummy[SPV_PC_N]; static pset_t g_pss[POOL_SETS]; static int g_npss;
static VkWriteDescriptorSet *g_pw; static VkDescriptorImageInfo *g_pwi; static int g_pnw, g_pcw;   // the writes of one sync
static uint64_t g_pool_syncs, g_pool_fast, g_pool_writes, g_pool_ns, g_pool_made, g_pool_deferred, g_pool_int, g_pool_base_sw, g_pool_stale, g_pool_stale_ns;
static int64_t g_pool_held;   // bytes of the images the sets hold (vtex msize, counted once per texture)
static uint64_t pool_sync_mb(void) { static int64_t v = -1; if (v < 0) { const char *e = getenv("DM_VK_POOL_SYNC_MB"); v = e && atoll(e) > 0 ? atoll(e) : 32; } return (uint64_t)v << 20; }
static VkImageView pool_dummy_img(VkImageViewType vt, VkImageType it, uint32_t layers, VkImageCreateFlags fl) {   // a black 1x1 stand-in of a class
    VkImageCreateInfo ii = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, fl, it, VK_FORMAT_R8G8B8A8_UNORM, {1, 1, 1}, 1, layers, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
                            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
    VkImage img; VKCHECK(vkCreateImage(g_dev, &ii, NULL, &img));
    VkMemoryRequirements r; vkGetImageMemoryRequirements(g_dev, img, &r); VkDeviceMemory m = mem_alloc(r, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT); VKCHECK(vkBindImageMemory(g_dev, img, m, 0));
    VkCommandBuffer cb = once_begin(); to_general(cb, img, VK_IMAGE_ASPECT_COLOR_BIT);
    VkClearColorValue black = {{0, 0, 0, 1}}; VkImageSubresourceRange rr = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers}; vkCmdClearColorImage(cb, img, VK_IMAGE_LAYOUT_GENERAL, &black, 1, &rr); once_end(cb);
    VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, img, vt, VK_FORMAT_R8G8B8A8_UNORM, {0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers}};
    VkImageView v; VKCHECK(vkCreateImageView(g_dev, &vi, NULL, &v)); return v;
}
static void pool_write(const pset_t *ps, int binding, uint32_t elem, VkImageView view, VkSampler sm) {
    if (g_pnw == g_pcw) { g_pcw = g_pcw ? 2 * g_pcw : 1024; g_pw = realloc(g_pw, g_pcw * sizeof *g_pw); g_pwi = realloc(g_pwi, g_pcw * sizeof *g_pwi); }
    g_pwi[g_pnw] = (VkDescriptorImageInfo){sm, view, VK_IMAGE_LAYOUT_GENERAL};
    g_pw[g_pnw++] = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, ps->ds, (uint32_t)binding, elem, 1, sm ? VK_DESCRIPTOR_TYPE_SAMPLER : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE};
}
static void pool_flush(void) {
    for (int i = 0; i < g_pnw; i++) g_pw[i].pImageInfo = &g_pwi[i];   // the info array may have moved while it grew
    if (g_pnw) vkUpdateDescriptorSets(g_dev, (uint32_t)g_pnw, g_pw, 0, NULL);
    g_pool_writes += (uint64_t)g_pnw; g_pnw = 0;
}
static int pool_init(void) {   // the stand-ins and the descriptor pool (room for POOL_SETS sets)
    static int tried; if (tried) return g_pool_dp != VK_NULL_HANDLE; tried = 1;
    if (!pool_dsl()) return 0;
    uint32_t cap = pool_cap();
    g_pool_dummy[SPV_PC_2D] = g_dummy_view;
    g_pool_dummy[SPV_PC_2DA] = pool_dummy_img(VK_IMAGE_VIEW_TYPE_2D_ARRAY, VK_IMAGE_TYPE_2D, 1, 0);
    g_pool_dummy[SPV_PC_3D] = pool_dummy_img(VK_IMAGE_VIEW_TYPE_3D, VK_IMAGE_TYPE_3D, 1, 0);
    g_pool_dummy[SPV_PC_CUBE] = pool_dummy_img(VK_IMAGE_VIEW_TYPE_CUBE, VK_IMAGE_TYPE_2D, 6, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT);
    VkDescriptorPoolSize ps[2] = {{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, POOL_SETS * SPV_PC_N * cap}, {VK_DESCRIPTOR_TYPE_SAMPLER, POOL_SETS * SPV_POOL_SAMP_CAP}};
    VkDescriptorPoolCreateInfo pi = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT, POOL_SETS, 2, ps};
    VkResult r = vkCreateDescriptorPool(g_dev, &pi, NULL, &g_pool_dp);
    if (r != VK_SUCCESS) { LOG("[RED] vk: pool set descriptor pool failed (%d): draws of pool shaders are left out", (int)r); g_pool_dp = VK_NULL_HANDLE; return 0; }
    return 1;
}
static pset_t *pset_new(void) {   // a set filled with stand-ins: an ID past the registrations reads black, never an unwritten descriptor
    uint32_t cap = pool_cap(); pset_t *ps = &g_pss[g_npss];
    VkDescriptorSetAllocateInfo ai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, g_pool_dp, 1, &g_pool_dsl_};
    VkResult r = vkAllocateDescriptorSets(g_dev, &ai, &ps->ds);
    if (r != VK_SUCCESS) { LOG("[RED] vk: pool set allocation failed (%d)", (int)r); return NULL; }
    uint64_t t0 = now_ns();
    for (int c = 0; c < SPV_PC_N; c++) {
        ps->view[c] = malloc(cap * sizeof *ps->view[c]);
        VkDescriptorImageInfo *ii = malloc(cap * sizeof *ii);
        for (uint32_t k = 0; k < cap; k++) { ps->view[c][k] = g_pool_dummy[c]; ii[k] = (VkDescriptorImageInfo){VK_NULL_HANDLE, g_pool_dummy[c], VK_IMAGE_LAYOUT_GENERAL}; }
        VkWriteDescriptorSet w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, ps->ds, (uint32_t)c, 0, cap, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, ii};
        vkUpdateDescriptorSets(g_dev, 1, &w, 0, NULL); free(ii);
    }
    ps->samp = malloc(SPV_POOL_SAMP_CAP * sizeof *ps->samp);
    { VkDescriptorImageInfo ii[SPV_POOL_SAMP_CAP]; for (int k = 0; k < SPV_POOL_SAMP_CAP; k++) { ps->samp[k] = g_dummy_samp; ii[k] = (VkDescriptorImageInfo){g_dummy_samp, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL}; }
      VkWriteDescriptorSet w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, ps->ds, SPV_POOL_SAMP, 0, SPV_POOL_SAMP_CAP, VK_DESCRIPTOR_TYPE_SAMPLER, ii}; vkUpdateDescriptorSets(g_dev, 1, &w, 0, NULL); }
    ps->v = calloc(cap, sizeof *ps->v);
    lk_(&g_rmu, LW_RMU); g_npss++; g_pool_live = g_npss; pthread_mutex_unlock(&g_rmu);   // registrations from now on are marked for it too (pool_mark)
    LOG("vk: pool set %d made: %u IDs x %d classes + %d samplers, filled with stand-ins in %.1f ms; %d shaders read the pool sets so far (%d left as before)",
        g_npss, cap, SPV_PC_N, SPV_POOL_SAMP_CAP, (now_ns() - t0) / 1e6, g_pool_shaders, g_pool_refused);
    return ps;
}
// The class a registered texture's view has (spv_pool.h SPV_PC_*), -1 when no binding of the set can hold it
static int pset_class(const treg_t *tr) {
    int target = tr->has_v && tr->v.set_target ? tr->v.target : tr->t.d.target;
    switch (target) {
    case NVN_TEXTURE_TARGET_2D: case NVN_TEXTURE_TARGET_RECTANGLE: return SPV_PC_2D;
    case NVN_TEXTURE_TARGET_2D_ARRAY: return SPV_PC_2DA;
    case NVN_TEXTURE_TARGET_3D: return SPV_PC_3D;
    case NVN_TEXTURE_TARGET_CUBEMAP: return SPV_PC_CUBE;
    default: return -1;
    }
}
static int vk_fmt_int(VkFormat f) {   // an integer format: a float-sampled binding cannot hold its view
    switch (f) {
    case VK_FORMAT_R8_UINT: case VK_FORMAT_R8_SINT: case VK_FORMAT_R8G8_UINT: case VK_FORMAT_R8G8_SINT: case VK_FORMAT_R8G8B8A8_UINT: case VK_FORMAT_R8G8B8A8_SINT:
    case VK_FORMAT_B8G8R8A8_UINT: case VK_FORMAT_B8G8R8A8_SINT: case VK_FORMAT_A2B10G10R10_UINT_PACK32: case VK_FORMAT_A2R10G10B10_UINT_PACK32:
    case VK_FORMAT_R16_UINT: case VK_FORMAT_R16_SINT: case VK_FORMAT_R16G16_UINT: case VK_FORMAT_R16G16_SINT: case VK_FORMAT_R16G16B16A16_UINT: case VK_FORMAT_R16G16B16A16_SINT:
    case VK_FORMAT_R32_UINT: case VK_FORMAT_R32_SINT: case VK_FORMAT_R32G32_UINT: case VK_FORMAT_R32G32_SINT: case VK_FORMAT_R32G32B32_UINT: case VK_FORMAT_R32G32B32_SINT:
    case VK_FORMAT_R32G32B32A32_UINT: case VK_FORMAT_R32G32B32A32_SINT: case VK_FORMAT_S8_UINT: return 1;
    default: return 0;
    }
}
static void pset_hold(pset_t *ps, uint32_t k, vtex *nv) {   // entry k now holds nv (or nothing): the counts eviction and freeing look at
    vtex *old = ps->v[k]; if (old == nv) return;
    if (old) { __atomic_store_n(&old->pool_out, g_fence + 1, __ATOMIC_RELEASE); if (!__atomic_sub_fetch(&old->inpool, 1, __ATOMIC_ACQ_REL)) g_pool_held -= (int64_t)old->msize; }   // freed after the submission being recorded
    if (nv && __atomic_add_fetch(&nv->inpool, 1, __ATOMIC_ACQ_REL) == 1) g_pool_held += (int64_t)nv->msize;
    ps->v[k] = nv;
}
// Entry k (host texture ID tb + k) as the registrations hold it now: the view of its texture in its class's binding (made and uploaded as a slot's would be), stand-ins elsewhere.
// A texture not made yet counts against *budget (bytes of storage, DM_VK_POOL_SYNC_MB a sync): past it the entry is left as it is and 0 returned (resolved at a later sync)
static int pool_resolve(pset_t *ps, uint32_t k, int64_t *budget) {
    int hid = ps->tb + (int)k; treg_t tr = {0}; vtex *have = NULL;
    if (hid > 0) { lk_(&g_rmu, LW_RMU); if (hid < g_ntreg) { tr = g_treg[hid]; if (tr.ok) have = tex_find(tr.t.serial); } pthread_mutex_unlock(&g_rmu); }
    vtex *nv = NULL; VkImageView view = VK_NULL_HANDLE; int cls = tr.ok ? pset_class(&tr) : -1;
    int stencil = tr.has_v && tr.v.set_dsmode ? tr.v.dsmode == NVN_TEXTURE_DEPTH_STENCIL_MODE_STENCIL : tr.t.stencil;
    if (cls >= 0 && !stencil) {
        if (!have || !have->img) { if (*budget <= 0) { g_pool_deferred++; return 0; } *budget -= (int64_t)(tr.t.L.size ? tr.t.L.size : 65536); }   // made (or brought back) here: its storage's size
        vtex *v = tex_get(&tr.t, 0); g_pool_made += v && v != have;
        if (v && v->img) { VkImageView tv = tex_view(v, &tr.t, &tr.v, tr.has_v, -1, 0); if (tv && vk_fmt_int(g_tv_fmt)) g_pool_int++; else if (tv) { view = tv; nv = v; } }
    }
    for (int c = 0; c < SPV_PC_N; c++) {
        VkImageView want = c == cls && view ? view : g_pool_dummy[c];
        if (ps->view[c][k] != want) { ps->view[c][k] = want; pool_write(ps, c, k, want, VK_NULL_HANDLE); }
    }
    pset_hold(ps, k, nv);
    return 1;
}
// Whether a held texture needs what a slot's use would do first (tex_slot's fast-path test): CPU-written, dirty pages in tracked storage, made before its storage was filled
static int pool_stale(const vtex *v, uint64_t pres) {
    return v->evicted || v->dirty || v->late || (v->lcpu && !(v->span && lcpu_dirty_on()) && v->lhash_gen != g_bvgen) ||
           (!v->gpu && v->span && v->clean_gen != g_bvgen && !(v->reup_frame && v->reup_frame == pres && !v->lcpu));
}
// The pool set for this draw's bases, brought up to date: everything at its first sync, else only what changed since its last one; once a frame its held textures are checked
// as slots' uses would check them (pool_stale)
static VkDescriptorSet pool_set(void) {
    if (!pool_init()) return VK_NULL_HANDLE;
    int tb = S.tex_base, sb = S.samp_base; uint32_t cap = pool_cap();
    pset_t *ps = NULL; for (int i = 0; i < g_npss; i++) if (g_pss[i].tb == tb && g_pss[i].sb == sb) ps = &g_pss[i];
    if (!ps) {
        if (g_npss < POOL_SETS) { if (!(ps = pset_new())) return VK_NULL_HANDLE; }
        else {   // the least recently used set the GPU is done with takes the new bases (resolved anew)
            uint64_t done; lk_(&g_mu, LW_MU); done = g_done; pthread_mutex_unlock(&g_mu);
            for (int i = 0; i < g_npss; i++) if (g_pss[i].last_fence <= done && (!ps || g_pss[i].used < ps->used)) ps = &g_pss[i];
            if (!ps) { RED_ONCE(90, "pool sets: more than %d pool bases in flight: a draw of a pool shader is left out", POOL_SETS); return VK_NULL_HANDLE; }
        }
        if (ps->synced) g_pool_base_sw++;
        ps->tb = tb; ps->sb = sb; ps->synced = 0;
    }
    ps->used = ++g_pool_syncs; ps->last_fence = g_fence + 1;
    uint64_t reg = atomic_load(&g_reg_gen), kill = atomic_load(&g_pool_kill), cpu = atomic_load(&g_pool_cpu), sgen = atomic_load(&g_pool_sgen), pres = atomic_load(&g_presents);
    int full = !ps->synced;
    if (!full && !ps->pending && reg == ps->reg && kill == ps->kill && cpu == ps->cpu && sgen == ps->sgen && pres == ps->pres) { g_pool_fast++; return ps->ds; }
    int slot = (int)(ps - g_pss);
    uint64_t t0 = now_ns(); int nres = 0, ndef = 0; int64_t budget = (int64_t)pool_sync_mb();
    static int *def; static int cdef; int ndf = 0;   // IDs left for a later sync (budget)
    #define DEFER(k_) do { if (ndf == cdef) { cdef = cdef ? 2 * cdef : 1024; def = realloc(def, cdef * sizeof *def); } def[ndf++] = (int)(k_); } while (0)
    if (full) {
        lk_(&g_rmu, LW_RMU); if (g_pool_dbits[slot]) memset(g_pool_dbits[slot], 0, g_pool_ndw[slot] * sizeof **g_pool_dbits); int nt = g_ntreg; pthread_mutex_unlock(&g_rmu);
        for (uint32_t k = 0; k < cap; k++) {
            if (tb + (int)k >= nt && !ps->v[k]) { int any = 0; for (int c = 0; c < SPV_PC_N; c++) any |= ps->view[c][k] != g_pool_dummy[c]; if (!any) continue; }   // past the registrations: a stand-in already
            if (pool_resolve(ps, k, &budget)) nres++; else DEFER(k);
        }
        int tn = atomic_load(&g_tpool_n); if (tn > tb + (int)cap) { static int nl; if (nl++ < 4) LOG("[RED] vk: pool set: the texture pools hold %d IDs, the set %u from base %d: the IDs past it read black (DM_VK_POOL_CAP)", tn, cap, tb); }
    } else {
        if (reg != ps->reg || ps->pending) {   // the IDs registered since, and those a budget left
            static int *ids; static int cids; int nids = 0;
            lk_(&g_rmu, LW_RMU);
            for (int w = 0; w < g_pool_ndw[slot]; w++) for (uint64_t m = g_pool_dbits[slot][w]; m; m &= m - 1) {
                int id = w * 64 + __builtin_ctzll(m) - tb; if (id < 0 || id >= (int)cap) continue;
                if (nids == cids) { cids = cids ? 2 * cids : 1024; ids = realloc(ids, cids * sizeof *ids); } ids[nids++] = id; }
            if (g_pool_dbits[slot]) memset(g_pool_dbits[slot], 0, g_pool_ndw[slot] * sizeof **g_pool_dbits);
            pthread_mutex_unlock(&g_rmu);
            for (int i = 0; i < nids; i++) { if (pool_resolve(ps, (uint32_t)ids[i], &budget)) nres++; else DEFER(ids[i]); }
        }
        if (kill != ps->kill) for (uint32_t k = 0; k < cap; k++) if (ps->v[k] && ps->v[k]->dead) { if (pool_resolve(ps, k, &budget)) nres++; else DEFER(k); }   // released while held
        if (cpu != ps->cpu || pres != ps->pres) {   // once a frame (and after a CPU write): held textures a slot's use would refresh, refreshed the same way
            uint64_t s0 = now_ns(); int nst = 0;
            for (uint32_t k = 0; k < cap; k++) { vtex *v = ps->v[k]; if (!v || !pool_stale(v, pres)) continue;
                int hid = tb + (int)k; treg_t tr = {0}; lk_(&g_rmu, LW_RMU); if (hid < g_ntreg) tr = g_treg[hid]; pthread_mutex_unlock(&g_rmu);
                if (tr.ok && tr.t.serial == v->serial) { vtex *n = tex_get(&tr.t, 0); if (n != v || !n || !n->img) pool_resolve(ps, k, &budget); else v->last_frame = pres; }
                else if (!pool_resolve(ps, k, &budget)) DEFER(k);
                nst++; }
            g_pool_stale += (uint64_t)nst; g_pool_stale_ns += now_ns() - s0; nres += nst;
        }
    }
    if (full || sgen != ps->sgen) {
        lk_(&g_rmu, LW_RMU);
        for (int k = 0; k < SPV_POOL_SAMP_CAP; k++) { VkSampler sm = sb + k > 0 ? samp_get(sb + k) : g_dummy_samp; if (ps->samp[k] != sm) { ps->samp[k] = sm; pool_write(ps, SPV_POOL_SAMP, (uint32_t)k, VK_NULL_HANDLE, sm); } }
        pthread_mutex_unlock(&g_rmu);
    }
    if (ndf) { lk_(&g_rmu, LW_RMU); for (int i = 0; i < ndf; i++) pool_mark_set(slot, tb + def[i]); pthread_mutex_unlock(&g_rmu); }   // the next sync takes them
    #undef DEFER
    ndef = ndf; ps->pending = ndf > 0;
    int nw = g_pnw; pool_flush();
    ps->synced = 1; ps->reg = reg; ps->kill = kill; ps->cpu = cpu; ps->sgen = sgen; ps->pres = pres;
    uint64_t dt = now_ns() - t0; g_pool_ns += dt;
    static uint64_t nlog; if (nlog++ < 8 || nlog % 1000 == 0 || dt > 20000000)
        LOG("vk: pool set %d (bases %d / %d) sync%s: %d IDs resolved, %d left for later (%d MB a sync), %d entries written, %.2f ms; in all: %llu syncs (%llu without a change), %llu entries, "
            "%.0f ms, textures made %llu, held %.0f MB, integer-format textures left as stand-ins %llu, base changes %llu, stale held textures refreshed %llu (%.0f ms)",
            slot + 1, tb, sb, full ? " (all IDs)" : "", nres, ndef, (int)(pool_sync_mb() >> 20), nw, dt / 1e6, (unsigned long long)g_pool_syncs, (unsigned long long)g_pool_fast,
            (unsigned long long)g_pool_writes, g_pool_ns / 1e6, (unsigned long long)g_pool_made, g_pool_held / 1048576.0, (unsigned long long)g_pool_int, (unsigned long long)g_pool_base_sw,
            (unsigned long long)g_pool_stale, g_pool_stale_ns / 1e6);
    return ps->ds;
}
// A held pass (merged across render-target changes) a pool shader's draw may read from: one of its written attachments is a texture a pool set holds (shader_reads_rt)
static int pool_reads_rt(void) {
    for (int i = 0; i < P.n; i++) if (P.ct[i] && P.cdrawn[i] && P.ct[i]->inpool) return 1;
    return P.dt && P.ddrawn && P.dt->inpool;
}
static VkDescriptorSet write_set_(shader_t *s, int stage) {
    if (stage == 5) g_nwb = 0;
    g_ndyno = 0;
    int reuse = stage_reuse() && stage < 5 && !g_plog && !s->dyn && !(stage == 1 && g_fb_n);
    if (reuse) { VkDescriptorSet r = wsc_try(s, stage); if (r) return r; g_wsc_miss++; }
    wsc_t nc = {0}; int fit = reuse;   // what this set is built from, if it can be reused so
    VkDescriptorSet ds = VK_NULL_HANDLE;
    enum { NE = 4096 };
    static VkWriteDescriptorSet w[64]; static VkDescriptorBufferInfo bi[NE]; static VkDescriptorImageInfo ii[NE]; static VkBufferView bv[NE]; int nw = 0, ne = 0;
    for (int i = 0; i < s->nb; i++) {
        const sbind *b = &s->b[i];
        uint32_t cnt = b->count ? b->count : 1;
        if (ne + (int)cnt > NE) { RED_ONCE(72, "shader %s: more than %d descriptors in one set: the rest are left unwritten", s->key, NE); break; }
        VkWriteDescriptorSet *x = &w[nw]; *x = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, ds, b->binding, 0, cnt, dtype(b->type)};
        if (b->type == BT_UBO || b->type == BT_SSBO) {
            for (uint32_t e = 0; e < cnt; e++) bi[ne + e] = (VkDescriptorBufferInfo){b->type == BT_SSBO ? g_scratch_buf : g_zero_buf, 0, 65536};
            VkDescriptorBufferInfo *d = &bi[ne];
            if (b->kind == K_CBUF && b->reg >= 0 && b->reg < 16) {
                const p_bind_buf *u = &S.ubo[stage][b->reg];
                uint64_t n = u->size > 65536 ? 65536 : u->size;
                VkBuffer gb; VkDeviceSize go;
                g_h0_caller = H0C_UBO; if (u->addr && n && gw_find(u->addr, n, g_props.limits.minUniformBufferOffsetAlignment ? g_props.limits.minUniformBufferOffsetAlignment : 256, &gb, &go, NULL)) *d = (VkDescriptorBufferInfo){gb, go, n};
                else if (u->addr && n) { VkDeviceSize o = copy_in(u->addr, n, 256, CI_UBO); if (o != VK_WHOLE_SIZE) *d = (VkDescriptorBufferInfo){C->ring, o, n}; }
            } else if (b->kind == K_C1 && s->c1_len) { VkBuffer rb; VkDeviceSize o = ring_put_memo(s->c1, (uint32_t)s->c1_len, 256, &rb); if (o != VK_WHOLE_SIZE) *d = (VkDescriptorBufferInfo){rb, o, s->c1_len}; }
            else if (b->kind == K_SUPPORT) *d = (VkDescriptorBufferInfo){g_support_buf, 0, 4096};
            else if ((b->kind == K_SBUF || b->kind == K_UBOP) && b->reg >= 0 && b->reg < 16) {   // a copy, or the GPU's own earlier write; a dispatch's are written back (dispatch)
                const p_bind_buf *u = b->kind == K_SBUF ? &S.ssbo[stage][b->reg] : &S.ubo[stage][b->reg];
                VkBuffer gb = VK_NULL_HANDLE; VkDeviceSize go = 0; uint8_t *gp = NULL; int gci = g_ci;
                VkDeviceSize al = b->type == BT_UBO ? g_props.limits.minUniformBufferOffsetAlignment : g_props.limits.minStorageBufferOffsetAlignment; if (!al) al = 256;
                int from_gw = 0;
                g_h0_caller = b->type == BT_UBO ? H0C_UBO : H0C_SSBO; if (u->addr && u->size && gw_find(u->addr, u->size, al, &gb, &go, &gp)) { *d = (VkDescriptorBufferInfo){gb, go, u->size}; gci = g_gw_ci; from_gw = 1; }
                else if (u->addr && u->size) { VkDeviceSize o = copy_in(u->addr, u->size, 256, b->type == BT_UBO ? CI_UBO : CI_SSBO); if (o != VK_WHOLE_SIZE) { *d = (VkDescriptorBufferInfo){C->ring, o, u->size}; gb = C->ring; go = o; gp = C->ringp + o; } }
                // A compute shader's storage buffer is what its dispatch may write: only a storage-buffer binding of a module that stores to one (a uniform binding, or a module
                // that only reads, writes nothing, and its range used to be written back whole at completion over whatever the CPU had written there since)
                if (stage == 5 && gb && b->type == BT_SSBO && s->ssbo_wr && g_nwb < 16) g_wb[g_nwb++] = (gw_t){u->addr, u->size, gb, go, gp, (uint64_t)from_gw, gci};
                if (stage != 5 && s->ssbo_wr) { static const shader_t *seen[8]; static int nseen; int k = 0; while (k < nseen && seen[k] != s) k++;
                                  if (k == nseen && nseen < 8) { seen[nseen++] = s; LOG("[RED] vk: shader %s (stage %d) binds storage buffer c%d at %#llx size %llu: what a draw writes to it is lost", s->key, stage, b->reg, (unsigned long long)u->addr, (unsigned long long)u->size); } }
            } else if (b->kind == K_DRVCB) {   // driver segments: c2 holds the bound texture / sampler handles (as nvn_mtl.m fills them), c0 the storage buffer table
                int skind = b->off == 2 ? (s->dyn ? 0 : 1) : 2;   // the segment's kind: what it is built from
                if (seg_cache_on() && stage < 6 && g_segc[stage][skind].gen == g_bvgen && g_segc[stage][skind].bgen == g_bgen[stage]) {   // built from the same bindings in this submission
                    *d = (VkDescriptorBufferInfo){g_segc[stage][skind].b, g_segc[stage][skind].off, 4096}; g_segc_hit++;
                } else {
                uint8_t seg[4096]; memset(seg, 0, sizeof seg);
                if (b->off == 2 && s->dyn) {   // tex_dyn_slots (as nvn_mtl.m): the tables hold slot indices, which the shader's arithmetic turns into indices of its slot arrays (dyn_slot)
                    for (int r = 0; r < 32; r++) { uint64_t h = S.tex[stage][r]; uint32_t v = !h ? 0 : (uint32_t)r | (h >> 20 ? (uint32_t)r << 20 : 0); memcpy(seg + 0x20 + 8 * r, &v, 4); }
                    for (int r = 0; r < 8; r++) { uint32_t v = (uint32_t)S.img[stage][r]; memcpy(seg + 0x120 + 8 * r, &v, 4); }
                    for (int r = 0; r < 128 && 0x168 + 8 * r + 4 <= 0x568; r++) { uint32_t v = S.tex[stage][128 + r] ? (uint32_t)r : 0; memcpy(seg + 0x168 + 8 * r, &v, 4); }
                    for (int r = 0; r < 128 && 0x568 + 8 * r + 4 <= (int)sizeof seg; r++) { uint32_t v = S.samp[stage][r] ? (uint32_t)r << 20 : 0; memcpy(seg + 0x568 + 8 * r, &v, 4); }
                } else if (b->off == 2) {
                    for (int r = 0; r < 32 && 0x20 + 8 * r < 0x120; r++) { uint32_t v = (uint32_t)S.tex[stage][r]; memcpy(seg + 0x20 + 8 * r, &v, 4); }
                    for (int r = 0; r < 8; r++) { uint32_t v = (uint32_t)S.img[stage][r]; memcpy(seg + 0x120 + 8 * r, &v, 4); }
                    for (int r = 0; r < 128 && 0x168 + 8 * r + 4 <= 0x568; r++) { uint32_t v = (uint32_t)S.tex[stage][128 + r]; memcpy(seg + 0x168 + 8 * r, &v, 4); }
                    for (int r = 0; r < 128 && 0x568 + 8 * r + 4 <= (int)sizeof seg; r++) { uint32_t v = (uint32_t)S.samp[stage][r]; memcpy(seg + 0x568 + 8 * r, &v, 4); }
                } else {
                    int base = stage == 0 ? 0x110 : stage == 1 ? 0x510 : stage == 5 ? 0x310 : -1;
                    if (base >= 0) for (int r = 0; r < 16; r++) { uint64_t a = S.ssbo[stage][r].addr; uint32_t n = (uint32_t)S.ssbo[stage][r].size; memcpy(seg + base + 16 * r, &a, 8); memcpy(seg + base + 16 * r + 8, &n, 4); }
                    if (stage != 5) { const float mp = 2047.0f; memcpy(seg + 0x610, &mp, 4); }
                }
                VkBuffer rb; VkDeviceSize o = ring_put_memo(seg, sizeof seg, 256, &rb); if (o != VK_WHOLE_SIZE) *d = (VkDescriptorBufferInfo){rb, o, sizeof seg};
                if (o != VK_WHOLE_SIZE && stage < 6) { g_segc[stage][skind].gen = g_bvgen; g_segc[stage][skind].bgen = g_bgen[stage]; g_segc[stage][skind].b = rb; g_segc[stage][skind].off = o; }
                g_segc_miss++;
                }
            }
            x->pBufferInfo = d;
            if (fit) { if (s->dynu[i] && cnt == 1 && (b->kind == K_CBUF || b->kind == K_C1 || b->kind == K_DRVCB) && nc.nd < 16) { nc.di[nc.nd] = (uint16_t)i; nc.db[nc.nd] = d->buffer; nc.dr[nc.nd] = d->range; nc.nd++; }
                       else if (!s->dynu[i] && cnt == 1 && (b->kind == K_SBUF || b->kind == K_UBOP) && b->reg >= 0 && b->reg < 16 && !s->ssbo_wr && nc.nck < 16)   // a storage (or pointer-bound uniform) buffer a draw stage only reads
                           nc.ck[nc.nck++] = (wck_t){1, (uint16_t)i, 0, VK_NULL_HANDLE, d->buffer, d->offset, d->range};
                       else if (s->dynu[i] || b->kind != K_SUPPORT) { fit = 0; g_wsc_nf[nc.nck >= 16 || nc.nd >= 16 ? WSC_NF_MANY : (b->kind == K_SBUF || b->kind == K_UBOP) ? WSC_NF_SSBO : WSC_NF_UBO]++; } }
            if (s->dynu[i]) { x->descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC; g_dyno[g_ndyno++] = (uint32_t)d->offset; d->offset = 0; }   // the offset goes with the bind
        } else if (b->type == BT_CIS || b->type == BT_TEX || b->type == BT_SAMP || b->type == BT_SIMG) {
            int simg = b->type == BT_SIMG; if (fit && (simg || b->skind == S_SAMP_UBO || b->kind == K_TEXU)) { fit = 0; g_wsc_nf[simg ? WSC_NF_SIMG : WSC_NF_TEXU]++; }   // handles in a uniform buffer change without a bind: never a reused set
            for (uint32_t e = 0; e < cnt; e++) {
                VkSampler sm = b->type == BT_TEX || simg ? VK_NULL_HANDLE : g_dummy_samp; VkImageView view = b->type == BT_SAMP ? VK_NULL_HANDLE : simg ? g_dummy_simg_view : g_dummy_view;
                uint64_t th = 0; int have = 0;
                int dyn = s->dyn && cnt > 1 && b->reg < 0;
                uint64_t tel = b->type != BT_SAMP && !simg && b->kind == K_TEXU && cnt > 1 ? texu_elem(stage, b, (int)e) : 0;   // an element of a handle array: 0 = no texture there (the stand-in)
                if (b->type != BT_SAMP && !simg && (dyn || (b->kind == K_TEX && b->reg >= 0 && b->reg + (int)e < NTEX) || (b->kind == K_TEXU && (cnt == 1 || tel)))) {
                    th = dyn ? dyn_slot(stage, (int)e) : b->kind == K_TEXU ? (cnt > 1 ? tel : texu_slot(stage, b)) : S.tex[stage][b->reg + e]; have = !dyn;
                    if (b->kind == K_TEXU && cnt == 1 && !(th & 0xfffff)) texu_zero_note(stage, s->key, b);   // (vk-t10 diagnostics)
                    int ok; vtex *tv; view = tex_slot(stage, th, &sm, &ok, &tv);
                    if (fit) { if (b->kind == K_TEXU) { fit = 0; g_wsc_nf[WSC_NF_TEXU]++; } else if (tv) { if (nc.nv < 64) nc.v[nc.nv++] = tv; else { fit = 0; g_wsc_nf[WSC_NF_MANY]++; } } }
                    if (stage == 0 && tv && tv->vhaz && tv->vhaz_epoch == g_vbar_epoch) g_vs_haz = 1;   // written by a pass whose end barrier left vertex shaders out (DM_VK_PASS_BARRIER)
                    if (tv && g_ndefer && tv->dfr_gen == g_defer_gen) g_defer_haz = 1;   // a storage image a deferred dispatch writes (defer_on)
                    if (stage == 5 && tv && P.open && ws_is_att(tv)) g_ws_att = 1;   // a dispatch sampling the open pass's attachment is not deferred
                    if (stage == 1 && tv) for (int q = 0; q < g_fb_n; q++) if (tv == g_fb_src[q]) { view = g_fb_view[q]; break; }   // same-pass feedback: the snapshot (fb_prepare)
                } else if (simg && b->kind == K_IMG && b->reg >= 0 && b->reg + (int)e < 32 && !(fit = 0)) {   // an image a shader writes: a single-level view with no swizzle, of an image made for storage
                    int tid = (int)(S.img[stage][b->reg + e] & 0xfffff);
                    lk_(&g_rmu, LW_RMU); treg_t tr = tid && tid < g_ntreg ? g_treg[tid] : (treg_t){0}; pthread_mutex_unlock(&g_rmu);
                    vtex *v = tr.ok && tr.t.d.target != NVN_TEXTURE_TARGET_BUFFER ? tex_get(&tr.t, 1) : NULL;
                    if (v && stage == 5) { if (P.open && ws_is_att(v)) g_ws_att = 1; if (g_nws_simg < 16) g_ws_simg[g_nws_simg++] = v; }   // what a deferred dispatch writes (defer_on)
                    else if (v && g_ndefer && v->dfr_gen == g_defer_gen) g_defer_haz = 1;
                    VkImageView iv = v && v->storage ? tex_view(v, &tr.t, &tr.v, tr.has_v, -1, 2) : VK_NULL_HANDLE;
                    if (iv) { view = iv; v->copied = 1; }   // written on the GPU: hibernation reads it back (mtl_hib_readback)
                    else RED_ONCE(73, "shader %s: image slot %d is not a storage-capable image here: its writes are lost", s->key, b->reg + (int)e);
                }
                if (b->type == BT_CIS && have) sm = bind_samp(stage, b, th, (int)e);
                else if (b->type == BT_CIS && dyn && (th >> 20)) { lk_(&g_rmu, LW_RMU); sm = samp_get((int)(th >> 20)); pthread_mutex_unlock(&g_rmu); }
                else if (b->type == BT_SAMP && dyn) { lk_(&g_rmu, LW_RMU); sm = samp_get((int)(S.samp[stage][e < 128 ? e : 0] >> 20)); pthread_mutex_unlock(&g_rmu); }   // sampler slot k
                else if (b->type == BT_SAMP && b->skind != S_NONE) sm = bind_samp(stage, b, 0, (int)e);
                ii[ne + e] = (VkDescriptorImageInfo){sm, view, VK_IMAGE_LAYOUT_GENERAL};
            }
            x->pImageInfo = &ii[ne];
        } else if (b->type == BT_TBUF || b->type == BT_UTB || b->type == BT_STB) {
            int st = b->type == BT_STB; if (fit && st) { fit = 0; g_wsc_nf[WSC_NF_STB]++; }
            for (uint32_t e = 0; e < cnt; e++) {
                VkBufferView v = VK_NULL_HANDLE; int r = b->reg >= 0 ? b->reg + (int)e : -1;
                if (s->dyn && cnt > 1 && b->reg < 0) { uint64_t h = dyn_slot(stage, (int)e); if (h) v = tbuf_view(stage, h, st); }   // slot k, when it holds a buffer texture
                else if (b->kind == K_TBUF && r >= 0 && r < NTEX) v = tbuf_view(stage, S.tex[stage][r], st);
                else if (b->kind == K_IMG && r >= 0 && r < 32) v = tbuf_view(stage, S.img[stage][r], st);
                if (fit) { if ((b->kind == K_TBUF || b->kind == K_IMG) && r >= 0 && nc.nck < 16) nc.ck[nc.nck++] = (wck_t){0, (uint16_t)i, (uint16_t)e, v};
                           else { fit = 0; g_wsc_nf[nc.nck >= 16 ? WSC_NF_MANY : WSC_NF_TB]++; } }
                bv[ne + e] = v ? v : st ? g_dummy_stb : g_dummy_utb; if (!v && r >= 0) g_ws_zero_tb++;
            }
            x->pTexelBufferView = &bv[ne];
        } else { RED_ONCE(22, "descriptor type %d not implemented yet (shader %s)", b->type, s->key); continue; }
        if (g_plog) {   // what each binding resolved to (DM_VK_PASSLOG / DM_VK_DUMP_FILE frames)
            C->plog = 1; char ln[512]; int o = snprintf(ln, sizeof ln, "plog:    %s b%u t%d k%d r%d", s->key, b->binding, b->type, b->kind, b->reg);
            if ((b->type == BT_UBO || b->type == BT_SSBO) && b->reg >= 0 && b->reg < 16 && (b->kind == K_CBUF || b->kind == K_SBUF || b->kind == K_UBOP)) {
                const p_bind_buf *u = b->kind == K_SBUF ? &S.ssbo[stage][b->reg] : &S.ubo[stage][b->reg]; float f[4] = {0};
                if (u->addr && u->size >= 16) memcpy(f, (const void *)(uintptr_t)u->addr, 16);
                o += snprintf(ln + o, sizeof ln - o, " buf %#llx +%llu %s guest %g %g %g %g", (unsigned long long)u->addr, (unsigned long long)u->size, bi[ne].buffer == C->ring ? "ring" : bi[ne].buffer == g_zero_buf || bi[ne].buffer == g_scratch_buf ? "zero" : "gpu-written", f[0], f[1], f[2], f[3]);
            } else if (b->type == BT_CIS || b->type == BT_TEX || b->type == BT_SIMG) {
                uint64_t h = b->type == BT_SIMG ? (b->reg >= 0 && b->reg < 32 ? S.img[stage][b->reg] : 0) : b->kind == K_TEXU ? texu_slot(stage, b) : b->reg >= 0 && b->reg < NTEX ? S.tex[stage][b->reg] : 0;
                int tid = (int)(h & 0xfffff); lk_(&g_rmu, LW_RMU); treg_t tr = tid && tid < g_ntreg ? g_treg[tid] : (treg_t){0}; pthread_mutex_unlock(&g_rmu);
                { vtex *pv = NULL; lk_(&g_rmu, LW_RMU); if (tr.ok) pv = tex_find(tr.t.serial); pthread_mutex_unlock(&g_rmu);
                  o += snprintf(ln + o, sizeof ln - o, " lv %u view %d+%d gpu %d gpu_lv %#x", tr.t.d.levels, tr.has_v && tr.v.set_levels ? tr.v.base_level : 0, tr.has_v && tr.v.set_levels ? tr.v.nlevels : 0,
                                pv ? pv->gpu : -1, pv ? pv->gpu_lv : 0);
                  int sid = (int)(h >> 20); if (b->type == BT_CIS && sid > 0 && sid < g_nsreg && g_sreg_ok[sid]) { const samp_par *sp = &g_sreg[sid]; o += snprintf(ln + o, sizeof ln - o, " samp min %d mag %d lod %g..%g bias %g cmp mode %d func %d red %d", sp->minf, sp->magf, sp->lod_min, sp->lod_max, sp->lod_bias, sp->cmp_mode, sp->cmp_func, sp->reduction); } }
                o += snprintf(ln + o, sizeof ln - o, " tex %llu %ux%u fmt %#x%s%s", tr.ok ? (unsigned long long)tr.t.serial : 0ull, tr.t.d.width, tr.t.d.height, tr.t.d.format,
                              ii[ne].imageView == g_dummy_view || ii[ne].imageView == g_dummy_simg_view ? " DUMMY" : "", tr.has_v && tr.v.set_dsmode && tr.v.dsmode ? " stencil" : "");
            } else if (b->type == BT_TBUF || b->type == BT_UTB || b->type == BT_STB) {
                int z = 0; for (uint32_t e = 0; e < cnt; e++) z += bv[ne + e] == g_dummy_utb || bv[ne + e] == g_dummy_stb;
                o += snprintf(ln + o, sizeof ln - o, " texel buffers %u, %d zero", cnt, z);
            }
            LOG("%s", ln);
        }
        ne += (int)cnt; nw++;
    }
    uint64_t key = 0xcbf29ce484222325ull, key2 = 0x84222325cbf29ce4ull; enum { KMAX = 64 }; uint64_t km[KMAX]; int nk = 0;
    #define HMIX(v) do { uint64_t v_ = (uint64_t)(v); key = (key ^ v_) * 0x100000001b3ull; key2 = (key2 + v_) * 0x9E3779B97F4A7C15ull; key2 ^= key2 >> 31; if (nk < KMAX) km[nk] = v_; nk++; } while (0)
    HMIX((uintptr_t)s->dsl);
    for (int k = 0; k < nw; k++) {
        const VkWriteDescriptorSet *x = &w[k]; HMIX(x->dstBinding | (uint64_t)x->descriptorCount << 32); HMIX(x->descriptorType);
        for (uint32_t e = 0; e < x->descriptorCount; e++) {
            if (x->pBufferInfo) { HMIX((uintptr_t)x->pBufferInfo[e].buffer); HMIX(x->pBufferInfo[e].offset); HMIX(x->pBufferInfo[e].range); }
            else if (x->pImageInfo) { HMIX((uintptr_t)x->pImageInfo[e].sampler); HMIX((uintptr_t)x->pImageInfo[e].imageView); }
            else if (x->pTexelBufferView) HMIX((uintptr_t)x->pTexelBufferView[e]);
        }
    }
    #undef HMIX
    typeof(g_dsc[0]) *ce = &g_dsc[key % 8192];
    if (ce->gen == g_bvgen && ce->key == key && ce->dsl == s->dsl) {
        if (!dset_cache2() || (ce->key2 == key2 && ce->nk == nk && (nk > KMAX || !memcmp(ce->km, km, (size_t)nk * 8)))) { g_dsc_hit++; wsc_keep(&nc, fit, s, stage, ce->ds); return ce->ds; }
        g_dsc_rej++;   // the same 64-bit hash, other bindings: a new set
    }
    static uint32_t nmiss; int timed = !tog(TG_LEAN) || nmiss++ % 7 == 0;   // DM_VK_LEAN: one miss in 7 timed (an odd period: misses alternate between stages; the line's average per set stays)
    uint64_t tm0 = timed ? stat_ns() : 0;
    VkDescriptorSetAllocateInfo ai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, C->dps[C->idp], 1, &s->dsl};
    while (vkAllocateDescriptorSets(g_dev, &ai, &ds) != VK_SUCCESS) {   // this pool is full: the next of the chain, made when there is none
        if (++C->idp == C->ndp) { if (C->ndp >= 256) { RED_ONCE(20, "256 descriptor pools in one submission: draws are dropped"); C->idp--; return VK_NULL_HANDLE; }
                                  C->dps = realloc(C->dps, (C->ndp + 1) * sizeof *C->dps); C->dps[C->ndp++] = dpool_new(); }
        ai.descriptorPool = C->dps[C->idp];
    }
    for (int k = 0; k < nw; k++) w[k].dstSet = ds;
    vkUpdateDescriptorSets(g_dev, (uint32_t)nw, w, 0, NULL);
    ce->key = key; ce->gen = g_bvgen; ce->dsl = s->dsl; ce->ds = ds; ce->key2 = key2; ce->nk = nk; if (nk <= KMAX) memcpy(ce->km, km, (size_t)nk * 8); g_dsc_miss++;
    if (timed) { g_dsc_miss_ns += stat_ns() - tm0; g_dsc_nmiss_t++; }
    wsc_keep(&nc, fit, s, stage, ds);
    return ds;
}
static _Atomic uint64_t g_wait_ns, g_rec_ns, g_pwait_ns;   // time the submit thread waited for the GPU, spent recording, and the present waited for its slot
static uint64_t g_stat_draws, g_stat_skip, g_skip_why[8];   // why: 0 no program, 1 shader missing / rejected, 2 no render target, 3 indirect, 4 primitive, 5 pipeline, 6 descriptors
#define SKIP(w) do { g_stat_skip++; g_skip_why[w]++; return; } while (0)
// ---------- Occlusion queries (NVN_COUNTER_TYPE_SAMPLES_PASSED) ----------
// NVN counts the samples that pass from ResetCounter on, and ReportCounter writes the count (and a timestamp) when the GPU completes. Here every draw made while the counter counts gets
// an occlusion query of its own (a query must begin and end inside one render pass), a reset starts a new list, and a report hands the list to the completion, which sums the results
// into the report. Until now every report said 0 samples: Unreal's hardware occlusion culling (its 36-index bounding boxes) took everything it tests as hidden, and FINAL FANTASY
// RESONANCE's field showed only its landscape -- no characters, trees or fires. The Metal host had the same bug once (nvn_mtl.m, visibility result buffers)
static int g_occ_on; static uint32_t *g_occ; static size_t g_nocc, g_cocc;
// The counter is the queue's, cumulative as on NVN: the draws counted since the last report or reset form a segment, which the completion adds in submission order (a reset sets it to 0,
// a report writes it out), so a game that reports before and after a query and subtracts sees the same as one that resets; draws after a submission's last report carry over (occ_flush)
static void occ_seg(uint64_t addr, int reset) {
    size_t n = g_occ_over ? SIZE_MAX : g_nocc; uint32_t *l = NULL;
    if (n != SIZE_MAX && n) { l = malloc(n * sizeof *l); memcpy(l, g_occ, n * sizeof *l); }
    if (n || addr || reset) post_add((post_t){4, NULL, (uint32_t)reset, 0, addr, 1, (const uint8_t *)l, n});
    g_nocc = 0; g_occ_over = 0;
}
static uint64_t g_occ_nrep, g_occ_nrst, g_occ_ndraw, g_occ_nq;   // the frame stats line
// DM_VK_OCC_MODE (vk19.44): 0 as before, a query around every draw from the first counter use on (the counter never stopped: ~937 of 1,001 draws a frame in the dungeon, against
// ~57 reset / report pairs); 1 the same, but counting only from a reset to the report after it; 2 (default) as 1, and one query for each stretch of counted draws in a pass (begun at
// the first, ended at the pass end, a report or a reset), as the Metal host keeps one visibility slot per pass. Counting stops after a report and starts again at a reset; a report
// that comes with no reset since the last one (a game that subtracts reports) switches to counting everything, as mode 0, for good
static int g_occ_q = -1, g_occ_cum;
static int occ_mode(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_OCC_MODE"); v = e ? atoi(e) : 2; if (v < 0 || v > 2) v = 2; LOG("vk: DM_VK_OCC_MODE=%d", v); } return v; }
static void occ_close(VkCommandBuffer cb) { if (g_occ_q >= 0) { vkCmdEndQuery(cb, C->oq, (uint32_t)g_occ_q); g_occ_q = -1; } }
static void occ_reset(void) { if (P.open) occ_close(C->cb); g_occ_on = 1; g_occ_nrst++; occ_seg(0, 1); }
static void occ_report(uint64_t addr) {
    if (P.open) occ_close(C->cb);
    if (occ_mode() && !g_occ_on && !g_occ_cum && g_occ_nrep) { g_occ_cum = 1; LOG("vk: occlusion: a report with no reset since the last one: every draw counted from now on (as DM_VK_OCC_MODE=0)"); }
    g_occ_nrep++; occ_seg(addr, 0);
    g_occ_on = !occ_mode() || g_occ_cum;   // paired: nothing counts again until the next reset
}
static void occ_flush(void) { if (g_occ_on) occ_seg(0, 0); }   // the end of a submission: its last segment, counted only
static int occ_begin(VkCommandBuffer cb) {   // the query of the draw about to be made, or -1
    if (!g_occ_on) return -1;
    if (occ_mode() == 2 && g_occ_q >= 0) { g_occ_ndraw++; return -1; }   // inside this stretch's query
    if (C->noq >= NOQ) { if (!g_occ_over) RED_ONCE(80, "more than %d occlusion-counted draws in one submission: the rest report as visible", NOQ); g_occ_over = 1; return -1; }
    uint32_t q = C->noq++;
    vkCmdBeginQuery(cb, C->oq, q, g_feat.occlusionQueryPrecise ? VK_QUERY_CONTROL_PRECISE_BIT : 0);
    if (g_nocc == g_cocc) { g_cocc = g_cocc ? 2 * g_cocc : 256; g_occ = realloc(g_occ, g_cocc * sizeof *g_occ); }
    g_occ[g_nocc++] = q; g_occ_ndraw++; g_occ_nq++;
    if (occ_mode() == 2) { g_occ_q = (int)q; return -1; }   // ended by occ_close
    return (int)q;
}
static void occ_end(VkCommandBuffer cb, int q) { if (q >= 0) vkCmdEndQuery(cb, C->oq, (uint32_t)q); }
// A draw that reads what a deferred dispatch writes (g_defer_haz, set while its resources were resolved): the pass ends, which records the deferred dispatches, and begins again;
// the pipeline, descriptor sets, vertex and index buffers and dynamic state bound stay bound across the two passes
#define DRAWQ(call) do { if (g_defer_haz && g_ndefer) { g_ndefer_haz++; pass_end(); if (!pass_begin()) return; } int q_ = occ_begin(cb); call; occ_end(cb, q_); } while (0)
// Indirect arguments, n bytes at addr: 1 when guest memory holds them and they were read into out (written by the CPU, or by a submission that has completed: gw_find waits for
// one still in flight), 2 when a dispatch of this submission writes them -- not run yet, so the GPU must read them: *b / *off is where
static int ind_args(uint64_t addr, uint32_t n, void *out, VkBuffer *b, VkDeviceSize *off, int gpu_ok) {
    g_h0_caller = H0C_IND; g_h0_indp = NULL; if (gw_find_(addr, n, 4, b, off, &g_h0_indp, gpu_ok)) return 2;   // gpu_ok (dispatches): an earlier submission's arguments are read by the GPU too; draws wait for them and read them here (no whole-stream copies)   // this submission's own: drawn by the GPU; an earlier one's is waited for and read here (no whole-stream copies)
    memcpy(out, (const void *)(uintptr_t)addr, n); return 1;
}
// The bytes of a pool from addr on (at most cap): the extent of an index buffer an indirect draw reads, whose count only the GPU will know
static uint64_t pool_left(uint64_t addr, uint64_t cap);
static uint64_t pool_base(uint64_t addr);
static uint64_t pool_range(uint64_t addr, uint64_t *size);
// DM_VK_IDXPOOL_ONCE (Harvest Moon heavy scene, H3 of docs/hm_android_perf_plan.md; off by default). An indexed draw whose indirect arguments the GPU writes copied the rest of
// its index pool from its own start (pool_left, at most 8 MB), keyed by that start: hm2 had 35 such draws a frame at 12 distinct starts in one pool, 76 MB of copies a frame
// (83% of copy_in's bytes). With it on, the first such draw of a submission copies the whole pool (at most DM_VK_IDXPOOL_ONCE_MB, default 32; a bigger pool: from the draw's
// start on) into the ring once, and the submission's later draws in that pool bind that copy at their own start, while the write log shows no write into it and the copy
// generation is the same. =2: each reuse also compares the copy with the guest's bytes the draw could read; differences are counted and logged, and the copy is made as before
static int idxpool_once_mode(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_IDXPOOL_ONCE"); v = e ? atoi(e) : 0; if (v < 0 || v > 2) v = 1; if (v) LOG("vk: DM_VK_IDXPOOL_ONCE=%d: the index pool of GPU-indirect indexed draws copied once a submission", v); } return v; }
static struct { uint64_t base, lo, hi, gen, fence; VkBuffer b; VkDeviceSize o; uint8_t *p; uint32_t seq; uint64_t hseq; } g_ipo[4]; static int g_ipo_next;
static uint64_t g_ipo_hit, g_ipo_copy, g_ipo_bytes, g_ipo_bad, g_ipo_fallback, g_ipo_hwl;
static uint64_t hwl_now(void); static int hwl_since(uint64_t seq, uint64_t lo, uint64_t hi);
static int ipo_get(uint64_t ibuf, uint64_t n, uint32_t isz, VkBuffer *b, VkDeviceSize *o) {
    uint64_t psize = 0, base = pool_range(ibuf, &psize); if (!base || !n) return 0;
    for (int i = 0; i < 4; i++) { typeof(g_ipo[0]) *e = &g_ipo[i];
        if (e->base != base || e->fence != g_fence + 1 || e->gen != g_wgen || ibuf < e->lo || ibuf + n > e->hi || e->seq > g_nwlog) continue;
        uint32_t k = e->seq; for (; k < g_nwlog; k++) if (g_wlo[k] < e->hi && g_whi[k] > e->lo) break;
        if (k != g_nwlog) { e->fence = 0; break; }   // written since: copied again
        if (hwl_since(e->hseq, ibuf, ibuf + n)) { e->fence = 0; g_ipo_hwl++; break; }   // a completed submission's write-back landed in it since: copied again
        if ((e->o + (ibuf - e->lo)) % isz) break;
        if (idxpool_once_mode() == 2 && memcmp(e->p + (ibuf - e->lo), (const void *)(uintptr_t)ibuf, n)) { static int nl; g_ipo_bad++;
            if (nl++ < 16) { const uint8_t *a = e->p + (ibuf - e->lo), *g = (const uint8_t *)(uintptr_t)ibuf; uint64_t first = UINT64_MAX, last = 0, pages = 0, lastpg = UINT64_MAX;
                for (uint64_t q = 0; q < n; q += 8) if (memcmp(a + q, g + q, n - q < 8 ? n - q : 8)) { if (first == UINT64_MAX) first = q; last = q; if (q >> 12 != lastpg) { lastpg = q >> 12; pages++; } }
                LOG("[RED] vk: DM_VK_IDXPOOL_ONCE=2: the pool copy differs from the guest's indices at %#llx +%llu: from +%llu to +%llu, %llu pages (pool %#llx, copy made %llu bytes before)",
                    (unsigned long long)ibuf, (unsigned long long)n, (unsigned long long)first, (unsigned long long)last, (unsigned long long)pages, (unsigned long long)e->base, (unsigned long long)(ibuf - e->lo)); }
            return 0; }
        *b = e->b; *o = e->o + (ibuf - e->lo); g_ipo_hit++; return 1;
    }
    static uint64_t cap; if (!cap) { const char *e = getenv("DM_VK_IDXPOOL_ONCE_MB"); long mb = e && atol(e) > 0 ? atol(e) : 32; cap = (uint64_t)mb << 20; }
    uint64_t lo = psize <= cap ? base : ibuf, hi = psize <= cap ? base + psize : (ibuf + cap < base + psize ? ibuf + cap : base + psize);
    if (hi - lo < n) { g_ipo_fallback++; return 0; }
    uint64_t hs = hwl_now();   // before the copy: a write-back during it counts as since
    VkDeviceSize ro = ring_put((const void *)(uintptr_t)lo, hi - lo, 16); if (ro == VK_WHOLE_SIZE) { g_ipo_fallback++; return 0; }
    if ((ro + (ibuf - lo)) % isz) { g_ipo_fallback++; return 0; }
    typeof(g_ipo[0]) *e = &g_ipo[g_ipo_next++ & 3]; *e = (typeof(g_ipo[0])){base, lo, hi, g_wgen, g_fence + 1, C->ring, ro, C->ringp + ro, g_nwlog, 0};
    e->hseq = hs;
    g_ipo_copy++; g_ipo_bytes += hi - lo;
    *b = C->ring; *o = ro + (ibuf - lo); return 1;
}
// H0: an indexed draw whose indirect arguments the GPU writes (its index-pool window copied from di->ibuf): distinct starts and pools a frame, summed over the window
static void h0_ipd(uint64_t ibuf, uint64_t n) {
    static uint64_t fr = UINT64_MAX, st[256], pl[32]; static int nst, npl;
    uint64_t f = atomic_load_explicit(&g_presents, memory_order_relaxed);
    if (f != fr) { fr = f; nst = npl = 0; }
    g_h0_ipd++; g_h0_ipd_bytes += n;
    int k = 0; for (; k < nst && st[k] != ibuf; k++) {} if (k == nst && nst < 256) { st[nst++] = ibuf; g_h0_ipd_starts++; }
    uint64_t b = pool_base(ibuf); for (k = 0; k < npl && pl[k] != b; k++) {} if (k == npl && npl < 32) { pl[npl++] = b; g_h0_ipd_pools++; }
}
// Live diagnostics: DM_VK_CTL_FILE=<path>, a file read while the game runs when it changed (vk_ctl; "set <switch> <value>": the run-time switches, see tog_set). "skip <first>-<last>" leaves out the draws with those numbers within each frame,
// "skipkey <shader key prefix>" the draws whose vertex or pixel shader key begins so; an empty or missing file leaves every draw in. For finding the draw that paints something
// wrong (FINAL FANTASY RESONANCE's blocky battle enemy) by bisection with screenshots, with no input to the game and no restart
static uint32_t g_frame_draw, g_skip_lo = 1, g_skip_hi; static char g_skip_key[8][17]; static int g_nskip_key;
static int ctl_skip(uint32_t didx, const char *vk, const char *fk) {
    if (didx >= g_skip_lo && didx <= g_skip_hi) return 1;
    for (int i = 0; i < g_nskip_key; i++) { size_t l = strlen(g_skip_key[i]); if ((vk && !strncmp(vk, g_skip_key[i], l)) || (fk && !strncmp(fk, g_skip_key[i], l))) return 1; }
    return 0;
}
// Read again when its inode, size, modification or change time differ (each frame: one stat), and in full every 30 frames whatever they say (vk19.59c): "set X 0" -> "set X 1"
// keeps the size, and shared storage may keep coarse times. Write it as a new file and rename it over the old (adb shell: echo ... > f.tmp && mv f.tmp f), so no read sees half of it
static void vk_ctl(uint64_t n) {
    static const char *path = (const char *)1; if (path == (const char *)1) path = getenv("DM_VK_CTL_FILE");
    if (!path) return;
    static struct timespec mt, ct; static off_t msz = -1; static ino_t ino; struct stat st;
    if (stat(path, &st)) { if (msz == -1) return; msz = -1; memset(&mt, 0, sizeof mt); }   // gone: as an empty file (every draw in; switches keep their values)
    else { if (n % 30 && st.st_ino == ino && st.st_mtim.tv_sec == mt.tv_sec && st.st_mtim.tv_nsec == mt.tv_nsec && st.st_ctim.tv_sec == ct.tv_sec && st.st_ctim.tv_nsec == ct.tv_nsec && st.st_size == msz) return;
           mt = st.st_mtim; ct = st.st_ctim; ino = st.st_ino; msz = st.st_size; }
    FILE *f = msz >= 0 ? fopen(path, "re") : NULL; uint32_t lo = 1, hi = 0; int nk = 0; char ln[160], k[17], nm[64];
    if (f) { while (fgets(ln, sizeof ln, f)) { unsigned a, b; int v;
        if (sscanf(ln, "skip %u-%u", &a, &b) == 2) { lo = a; hi = b; } else if (sscanf(ln, "skipkey %16s", k) == 1 && nk < 8) strcpy(g_skip_key[nk++], k);
        else if (sscanf(ln, "set %63s %d", nm, &v) == 2) tog_set(nm, v, n); } fclose(f); }
    if (lo != g_skip_lo || hi != g_skip_hi || nk != g_nskip_key) LOG("vk: control file: skipping draws %u-%u%s, %d shader key prefixes (frame %llu)", lo, hi, lo > hi ? " (none)" : "", nk, (unsigned long long)n);
    g_skip_lo = lo; g_skip_hi = hi; g_nskip_key = nk;
}
// A pass kept open across a SetRenderTargets (the hold) joins work the game drew into separate passes; a draw that samples one of the pass's attachments -- the scene depth a
// contact-shadow or ambient-occlusion term reads, a colour target -- would read it while the pass still holds it in tile memory (undefined: FINAL FANTASY RESONANCE's castle
// corridor, the contact shadow along the wall flickering frame to frame). Such a draw ends the pass first, as the Metal host does (nvn_mtl.m draw_samples_rt). Every texture
// binding of the draw's shaders is looked at: slots, handles from uniform buffers, dynamic slot arrays
// Only an attachment cleared or written in this pass counts: one the pass only loaded holds in memory what the tile holds, so sampling it reads the same and the pass goes on
// (the split stores and loads every attachment of the pass again, a full tile store and load on the Adreno). g_rmu is taken once for the whole draw. DM_VK_RT_SPLIT=0: no split
static uint64_t g_nsplit, g_nsplit_skip; static uint64_t g_t_split;
static int rt_split_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_RT_SPLIT"); v = !(e && !strcmp(e, "0")); if (!v) LOG("DM_VK_RT_SPLIT=0: a held pass is not ended for a draw that samples one of its attachments"); } return v; }
static int rt_serial_(uint64_t h, uint64_t *ser) {   // caller holds g_rmu
    int tid = (int)(h & 0xfffff); if (!tid) return 0;
    int ok = tid < g_ntreg && g_treg[tid].ok; if (ok) *ser = g_treg[tid].t.serial;
    return ok;
}
static int g_rt_unwritten;   // the draw samples an attachment the pass did not write (counted, no split)
static int is_attachment(uint64_t ser) {
    if (P.dt && P.dt->serial == ser) { if (P.ddrawn) return 1; g_rt_unwritten = 1; }
    for (int i = 0; i < P.n; i++) if (P.ct[i] && P.ct[i]->serial == ser) { if (P.cdrawn[i]) return 1; g_rt_unwritten = 1; }
    return 0;
}
static int shader_reads_rt(shader_t *s, int stage) {   // caller holds g_rmu
    if (!s) return 0;
    if (s->pool && pool_reads_rt()) return 1;   // vk_pool: it may sample any texture a pool set holds
    for (int i = 0; i < s->nb; i++) {
        const sbind *b = &s->b[i]; if (b->type != BT_CIS && b->type != BT_TEX) continue;
        uint32_t cnt = b->count ? b->count : 1; uint64_t ser;
        if (s->dyn && cnt > 1 && b->reg < 0) { for (uint32_t e = 0; e < cnt && e < 128; e++) if (rt_serial_(dyn_slot(stage, (int)e), &ser) && is_attachment(ser)) return 1; continue; }
        if (b->kind == K_TEXU && cnt == 1) { if (rt_serial_(texu_slot(stage, b), &ser) && is_attachment(ser)) return 1; continue; }
        if (b->kind == K_TEXU) { for (uint32_t e = 0; e < cnt; e++) if (rt_serial_(texu_elem_h(stage, b, (int)e), &ser) && is_attachment(ser)) return 1; continue; }   // an array of handles: every element
        if (b->kind == K_TEX && b->reg >= 0) for (uint32_t e = 0; e < cnt && b->reg + (int)e < NTEX; e++) if (rt_serial_(S.tex[stage][b->reg + e], &ser) && is_attachment(ser)) return 1;
    }
    return 0;
}
static int draw_reads_rt(shader_t *vs, shader_t *fs, shader_t *gs) {
    if (!rt_split_on()) return 0;
    uint64_t t0 = now_ns(); g_rt_unwritten = 0;
    lk_(&g_rmu, LW_RMU); int r = shader_reads_rt(vs, 0) || shader_reads_rt(fs, 1) || shader_reads_rt(gs, 2); pthread_mutex_unlock(&g_rmu);
    if (!r && g_rt_unwritten) g_nsplit_skip++;
    g_t_split += now_ns() - t0;
    return r;
}
// DM_VK_STATE_SKIP (vk19.41, default 1), after the Metal host's setters that skip unchanged state: the last draw's pipeline is taken again without hashing its key when the
// key and shaders are the same, and the pipeline bind and each piece of dynamic state (viewport, scissor, depth bias, blend constants, the stencil masks and references) are
// recorded only when they differ from what this command buffer last had. Reset at each submission's start (a new command buffer)
static struct { shader_t *vs, *fs, *gs; pso_key k; VkPipelineLayout pl; VkPipeline pp, bound; VkViewport vp; VkRect2D sc; float db[3], bc[4]; uint32_t st[2][3];
                int has_vp, has_sc, has_db, has_bc, has_st[2];
                VkBuffer vb[16]; VkDeviceSize vo[16]; uint32_t vb_ok;   // the vertex buffers bound in this command buffer (vb_ok: which slots)
                VkBuffer ib; int it; int ib_ok;                     // the index buffer bound (at offset 0: draws index it by firstIndex)
                VkPipelineLayout dpl; int dn; VkDescriptorSet dsets[2]; uint32_t ndy; uint32_t dy[192]; int ds_ok; } g_ds_last;
static uint64_t g_bind_skips;
static void ds_last_newcb(void) {   // nothing bound in the command buffer being recorded; the pipeline memo (by key) stays
    shader_t *v0 = g_ds_last.vs, *f0 = g_ds_last.fs, *g0 = g_ds_last.gs; pso_key k0 = g_ds_last.k; VkPipelineLayout l0 = g_ds_last.pl; VkPipeline p0 = g_ds_last.pp;
    memset(&g_ds_last, 0, sizeof g_ds_last); g_ds_last.vs = v0; g_ds_last.fs = f0; g_ds_last.gs = g0; g_ds_last.k = k0; g_ds_last.pl = l0; g_ds_last.pp = p0;
}
static uint64_t g_pso_memo, g_state_skips;
static int state_skip(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_STATE_SKIP"); v = !e || atoi(e) != 0; LOG("vk: DM_VK_STATE_SKIP=%d", v); } return v; }
static void draw(const p_draw *dw, const p_draw_idx *di) {
    if (!S.bst[0]) SKIP(0);
    g_defer_haz = 0;
    uint32_t didx = g_frame_draw++;   // this draw's number within the frame (the live control file's skip ranges)
    // Indirect draws: the record {count, instances, first, base instance} (indexed: {count, instances, first index, base vertex, base instance}). Read here when the CPU can, and
    // drawn as an ordinary draw (only the vertices it uses copied); written by a dispatch of this very submission (Unreal's GPU particles and culling), drawn by the GPU from the
    // ring with the whole vertex streams bound. An indirect-count multi-draw's record i is drawn while i < its count
    VkBuffer ind_b = VK_NULL_HANDLE; VkDeviceSize ind_o = 0;
    if ((dw && dw->ind) || (di && di->ind)) {
        uint64_t mc = dw ? dw->mdi_cnt : di->mdi_cnt; int mi = dw ? dw->mdi_i : di->mdi_i;
        if (mc) { uint32_t cnt = 0; VkBuffer cb_; VkDeviceSize co;
                  if (ind_args(mc, 4, &cnt, &cb_, &co, 0) == 2) RED_ONCE(35, "an indirect-count multi-draw's count is written by a dispatch of the same submission: each of its draws is made (their records may hold zeros)");
                  else if ((uint32_t)mi >= cnt) return; }
        uint64_t vb = 0; for (int s2 = 0; s2 < 16; s2++) if (S.vbuf[s2].address) vb += S.vbuf[s2].size;   // what a GPU-read draw copies (its whole streams)
        uint32_t a[5] = {0}; int k = ind_args(dw ? dw->ind : di->ind, dw ? 16 : 20, a, &ind_b, &ind_o, vb <= (8u << 20));   // small streams: a running submission's arguments are read by the GPU, not waited for
        if (k == 1) {
            if (!a[0] || !a[1]) { g_stat_draws++; return; }   // nothing drawn (0 instances is none, unlike a direct draw's 0)
            if (a[0] > (1u << 24) || a[1] > (1u << 20)) { RED_ONCE(36, "indirect draw of %u vertices x %u instances: not plausible, skipped", a[0], a[1]); SKIP(3); }
            if (dw) { p_draw t = *dw; t.count = (int32_t)a[0]; t.instances = (int32_t)a[1]; t.first = (int32_t)a[2]; t.base_instance = (int32_t)a[3]; t.ind = t.mdi_cnt = 0; draw(&t, NULL); }
            else { p_draw_idx t = *di; uint32_t isz = di->itype == 0 ? 1 : di->itype == 1 ? 2 : 4;
                   t.count = (int32_t)a[0]; t.instances = (int32_t)a[1]; t.ibuf = di->ibuf + (uint64_t)a[2] * isz; t.base_vertex = (int32_t)a[3]; t.base_instance = (int32_t)a[4]; t.ind = t.mdi_cnt = 0; draw(NULL, &t); }
            return;
        }
        static int once; if (!once++) LOG("vk: an indirect draw's arguments come from a dispatch of the same submission: drawn by the GPU (logged once)");
    }
    shader_t *vs = NULL, *fs = NULL, *gs = NULL;
    for (int st = 0; st < 3; st++) {   // the vertex shader of the program bound to the vertex stage, the fragment shader of the one bound to the fragment stage (cached by program and serial)
        const prog_t *p = S.bst[st]; if (!p) continue;
        static struct { const prog_t *p; uint64_t ser; shader_t *sh[3]; } pc[1024];
        unsigned h = (unsigned)(((uintptr_t)p >> 4) ^ p->serial) % 1024;
        if (pc[h].p != p || pc[h].ser != p->serial) {
            pc[h].p = p; pc[h].ser = p->serial; pc[h].sh[0] = pc[h].sh[1] = pc[h].sh[2] = NULL;
            for (unsigned i = 0; i < p->nstage; i++) { uint8_t md[8]; if (!prog_key(p, i, md)) continue; shader_t *s = shader_get(md); if (s->stage >= 0 && s->stage <= 2) pc[h].sh[s->stage] = s; }
        }
        if (st == 2) gs = pc[h].sh[2]; else if (st) fs = pc[h].sh[1]; else vs = pc[h].sh[0];
    }
    // A geometry stage (FINAL FANTASY XII THE ZODIAC AGE's character shaders: 84 geometry shaders, whose pixel shaders read a varying only the geometry shader writes) runs as
    // Vulkan's own, its resources in set 2, with the GPU's geometryShader. Without it, or for tessellation, the draw goes without those stages as before
    if (gs && (!g_feat.geometryShader || !gs->mod)) { RED_ONCE(33, "geometry stage: %s; drawn without it", !g_feat.geometryShader ? "the GPU has no geometry shaders" : "the shader was not loaded"); gs = NULL; }
    if (S.bst[3] || S.bst[4]) RED_ONCE(32, "tessellation stages are bound: not drawn with them yet");
    if (vk_skip() & SK_DRAW) SKIP(4);
    if (!vs || !vs->mod || (fs && !fs->mod)) SKIP(1);
    if ((g_skip_lo <= g_skip_hi || g_nskip_key) && ctl_skip(didx, vs->key, fs ? fs->key : NULL)) { if (g_plog) LOG("plog:  draw %u skipped (control file)", didx); return; }
    if (g_plog) LOG("plog:  draw number %u in the frame", didx);
    { int lay = vs->layer_out || (gs && gs->layer_out);   // a layered pass for a shader that writes the layer; a pass opened the other way is ended first
      if (P.open && ((lay && !P.layered && !(P.lok && P.nlayers <= 1)) || (!lay && P.layered))) pass_end();
      P.want_layered = lay; }
    if (P.open && P.merged && draw_reads_rt(vs, fs, gs)) { pass_end(); g_nsplit++; g_after_split = 1; }   // what was drawn into it is stored first, then read (as the pass the hold joined would have)
    if (!pass_begin()) SKIP(2);
    if (C->ntq & 1) C->tdraws[C->ntq / 2]++;   // draws in the timed pass (its signature's line)
    if (ind_b && (dw ? dw->prim : di->prim) == NVN_DRAW_PRIMITIVE_QUADS) { RED_ONCE(30, "a GPU-written indirect draw of quads is not implemented"); SKIP(3); }
    if (ind_b && di && di->itype == 0 && !g_has_u8) { RED_ONCE(37, "a GPU-written indirect draw with 8-bit indices needs VK_EXT_index_type_uint8"); SKIP(3); }
    int h0slot = -1;
    if (ind_b && g_h0_indp && C->nh0ip < 256 && h0_on()) { h0slot = C->nh0ip; C->h0ip[C->nh0ip] = g_h0_indp; C->h0ipp[h0slot] = NULL; C->h0iw[h0slot] = 0; C->h0vmax[h0slot] = UINT32_MAX; C->h0isz[h0slot] = 0;
                                                 C->h0ix[C->nh0ip++] = (uint8_t)((di ? 1 : 0) | (g_h0_cpass ? 2 : 0)); }   // H0: read back at completion
    int prim = dw ? dw->prim : di->prim;
    if (prim == NVN_DRAW_PRIMITIVE_LINE_LOOP || prim == NVN_DRAW_PRIMITIVE_QUAD_STRIP || prim == NVN_DRAW_PRIMITIVE_PATCHES) { RED_ONCE(31, "primitive %d not implemented yet", prim); SKIP(4); }
    pso_key k; memset(&k, 0, sizeof k);
    k.vs = vs; k.fs = fs; k.gs = gs; k.prim = prim; k.nattr = S.nattr; memcpy(k.attr, S.attr, sizeof k.attr); k.nstream = S.nstream; memcpy(k.stream, S.stream, sizeof k.stream);
    k.nc = P.n; for (int i = 0; i < P.n; i++) { k.cf[i] = P.ct[i] && P.cv[i] ? (P.cf[i] ? P.cf[i] : P.ct[i]->f.f) : VK_FORMAT_UNDEFINED; k.blend[i] = S.blend[i]; }
    k.color = S.color; k.chmask = S.chmask; k.poly = S.poly; k.samples = P.ct[0] ? 1 : 1;
    if (S.color.alpha_func != NVN_ALPHA_FUNC_ALWAYS) RED_ONCE(48, "the alpha test (ColorStateSetAlphaTest %d) is not implemented: fragments are not discarded by it", (int)S.color.alpha_func);
    // Lines wider than one pixel (SetLineWidth): FINAL FANTASY XII THE ZODIAC AGE draws its battle target lines 3 pixels wide. The Metal host draws them as quads (it has only
    // one-pixel lines, nvn_geometry.inc line_mesh_draw); Vulkan rasterizes them with wideLines, clamped to the GPU's range. Without the feature they stay one pixel wide
    if (prim == NVN_DRAW_PRIMITIVE_LINES || prim == NVN_DRAW_PRIMITIVE_LINE_STRIP || prim == NVN_DRAW_PRIMITIVE_LINE_LOOP) {
        float lw = S.line_w;
        if (lw > 1 && !g_feat.wideLines) { RED_ONCE(50, "lines %.1f pixels wide: the GPU has no wide lines, drawn one pixel wide", lw); lw = 1; }
        if (lw > g_props.limits.lineWidthRange[1]) lw = g_props.limits.lineWidthRange[1];
        k.line_w = lw > 1 ? lw : 0;
    }
    // Primitive restart: Vulkan restarts at the all-ones index only, and in lists only with VK_EXT_primitive_topology_list_restart
    if (S.restart.on && di) {
        uint32_t all = di->itype == 0 ? 0xff : di->itype == 1 ? 0xffff : 0xffffffffu;
        int strip = prim == NVN_DRAW_PRIMITIVE_LINE_STRIP || prim == NVN_DRAW_PRIMITIVE_TRIANGLE_STRIP || prim == NVN_DRAW_PRIMITIVE_TRIANGLE_FAN || prim == NVN_DRAW_PRIMITIVE_LINE_STRIP_ADJACENCY || prim == NVN_DRAW_PRIMITIVE_TRIANGLE_STRIP_ADJACENCY;
        if (S.restart.index != all) RED_ONCE(51, "primitive restart index %#x: Vulkan restarts at %#x only", S.restart.index, all);
        k.restart = strip;
    }
    // Dual-source blending (FINAL FANTASY XII THE ZODIAC AGE's dual_want shaders): with the GPU's dualSrcBlend, through the pixel shader's output 1 moved to index 1 (spv_dual)
    if (fs && g_feat.dualSrcBlend) for (int i = 0; i < P.n && !k.dual; i++)
        if ((S.color.blend_en >> i & 1) && (bf_src1(S.blend[i].src_rgb) || bf_src1(S.blend[i].dst_rgb) || bf_src1(S.blend[i].src_a) || bf_src1(S.blend[i].dst_a))) k.dual = fs_dual_mod(fs) != VK_NULL_HANDLE;
    if (P.dt) { k.df = P.dt->f.f; k.stencil = P.dt->f.stencil; k.ds = S.ds; }
    if (g_plog) LOG("plog:  draw vs %s fs %s n %u depth t%d w%d f%d stencil %d f%d ops %d/%d/%d ref %u vm %#x wm %#x blend %#x mask %#x cull %d", vs->key, fs ? fs->key : "-", dw ? dw->count : di->count,
                    S.ds.dt, S.ds.dw, S.ds.df, S.ds.st, S.ds.sf_f, S.ds.fail_f, S.ds.zfail_f, S.ds.zpass_f, S.st_ref[0], S.st_vmask[0], S.st_wmask[0], S.color.blend_en, S.chmask.mask, S.poly.cull);
    if (g_plog) {   // the vertex inputs the shader reads: attribute, stream, offset, NVN and Vulkan formats; the streams' strides and divisors (the dark vegetation capture)
        char ab[768]; int ao = 0;
        for (int i = 0; i < S.nattr && ao < (int)sizeof ab - 48; i++) if (vs->vin_mask >> i & 1) ao += snprintf(ab + ao, sizeof ab - ao, " a%d s%u +%u fmt %#x vk %d", i, (unsigned)S.attr[i].stream, (unsigned)S.attr[i].offset, (unsigned)S.attr[i].fmt, (int)vfmt(S.attr[i].fmt));
        for (int st = 0; st < S.nstream && st < 16 && ao < (int)sizeof ab - 40; st++) if (S.vbuf[st].address) ao += snprintf(ab + ao, sizeof ab - ao, " | s%d stride %u div %u bound %#llx +%llu", st, (unsigned)S.stream[st].stride, (unsigned)S.stream[st].divisor, (unsigned long long)S.vbuf[st].address, (unsigned long long)S.vbuf[st].size);
        LOG("plog:   vertex inputs%s%s", ab, ao ? "" : " none"); }
    VkPipelineLayout pl; VkPipeline pp;
    if (state_skip() && g_ds_last.vs == vs && g_ds_last.fs == fs && g_ds_last.gs == gs && g_ds_last.pp && !memcmp(&g_ds_last.k, &k, sizeof k)) { pl = g_ds_last.pl; pp = g_ds_last.pp; g_pso_memo++; }   // the last draw's pipeline (DM_VK_STATE_SKIP)
    else {
        pl = layout_get(vs, fs, gs);
        pp = pso_get(&k, pl, !(P.dt && P.n));   // the scene's passes (colour and depth) do not wait for a new pipeline
        if (pp && state_skip()) { g_ds_last.vs = vs; g_ds_last.fs = fs; g_ds_last.gs = gs; g_ds_last.k = k; g_ds_last.pl = pl; g_ds_last.pp = pp; }
    }
    if (!pp) { if (atomic_load(&g_pso_pending)) { g_stat_skip++; g_skip_why[7]++; return; } SKIP(5); }
    if (fs && P.n) fb_prepare(fs); else g_fb_n = 0;
    g_ws_zero_tb = 0;
    uint32_t dyno[192], ndyno = 0, gdyno[64], ngdyno = 0;
    g_vs_haz = 0;
    VkDescriptorSet sets[2]; sets[0] = write_set(vs, 0); memcpy(dyno, g_dyno, g_ndyno * 4); ndyno = g_ndyno;
    if (g_vs_haz) {   // draw_vs_hazard: the vertex shader samples what a pass wrote with vertex shaders left out of its end barrier: the open pass ends, the barrier, and it begins again
        int was = P.open; if (was) pass_end();
        VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
        vkCmdPipelineBarrier(C->cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
        g_vbar_epoch++; g_nbar_vs++;
        if (was && !pass_begin()) SKIP(2);
    }
    sets[1] = VK_NULL_HANDLE; if (fs) { sets[1] = write_set(fs, 1); memcpy(dyno + ndyno, g_dyno, g_ndyno * 4); ndyno += g_ndyno; }
    g_fb_n = 0;
    VkDescriptorSet gset = VK_NULL_HANDLE; if (gs) { gset = write_set(gs, 2); memcpy(gdyno, g_dyno, g_ndyno * 4); ngdyno = g_ndyno; }
    VkDescriptorSet pset = VK_NULL_HANDLE; if (fs && fs->pool && !(pset = pool_set())) SKIP(6);   // the pool set, kept up to date with the registrations (vk_pool)
    if (!sets[0] || (fs && !sets[1]) || (gs && !gset)) SKIP(6);
    for (int a = 0; a < P.n; a++) if (S.chmask.mask >> (4 * a) & 0xf) P.cwr[a] = P.cdrawn[a] = P.pw_c[a] = 1;
    if (P.dt && ((S.ds.dt && S.ds.dw) || S.ds.st)) P.ddrawn = P.pw_d = 1;   // depth writes, or a stencil test (its ops may write)
    VkCommandBuffer cb = C->cb;
    if (!state_skip() || g_ds_last.bound != pp) { vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pp); g_ds_last.bound = pp; } else g_state_skips++;
    {   // the same sets and offsets with the same layout as the last draw: left bound (DM_VK_STATE_SKIP)
        int dn = fs ? 2 : 1;
        if (state_skip() && !gs && g_ds_last.ds_ok && g_ds_last.dpl == pl && g_ds_last.dn == dn && g_ds_last.dsets[0] == sets[0] && (dn < 2 || g_ds_last.dsets[1] == sets[1]) && g_ds_last.ndy == ndyno && !memcmp(g_ds_last.dy, dyno, ndyno * 4)) g_bind_skips++;
        else {
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pl, 0, dn, sets, ndyno, dyno);
            if (state_skip() && !gs && ndyno <= 192) { g_ds_last.dpl = pl; g_ds_last.dn = dn; g_ds_last.dsets[0] = sets[0]; g_ds_last.dsets[1] = sets[1]; g_ds_last.ndy = ndyno; memcpy(g_ds_last.dy, dyno, ndyno * 4); g_ds_last.ds_ok = 1; } else g_ds_last.ds_ok = 0;
        }
    }
    if (gs) { vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pl, 2, 1, &gset, ngdyno, gdyno); g_ds_last.ds_ok = 0; }
    if (pset) vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pl, SPV_POOL_SET, 1, &pset, 0, NULL);
    // NVN's upper-left origin keeps D3D's y-up clip space; Vulkan's is y-down: the viewport is flipped (negative height)
    float x = 0, y = 0, w = (float)P.w, h = (float)P.h;
    if (S.has_vp) { x = S.vp[0][0]; y = S.vp[0][1]; w = S.vp[0][2]; h = S.vp[0][3]; }
    float yt = g_origin_ll ? (float)P.h - y - h : y;   // lower-left origin: NVN's y counts from the bottom
    VkViewport vv = {x, yt + h, w, -h, S.dr[0], S.dr[1] ? S.dr[1] : 1};
    int sk = state_skip();
    if (!sk || !g_ds_last.has_vp || memcmp(&g_ds_last.vp, &vv, sizeof vv)) { vkCmdSetViewport(cb, 0, 1, &vv); g_ds_last.vp = vv; g_ds_last.has_vp = 1; } else g_state_skips++;
    VkRect2D sc = {{0, 0}, {P.w, P.h}};
    if (S.has_sc) {
        int32_t sx = S.sc.x < 0 ? 0 : S.sc.x, sy = S.sc.y < 0 ? 0 : S.sc.y, ex = S.sc.x + S.sc.w, ey = S.sc.y + S.sc.h;
        if (g_origin_ll) { int32_t t = (int32_t)P.h - ey; ey = (int32_t)P.h - S.sc.y; sy = t < 0 ? 0 : t; }
        if (ex > (int32_t)P.w) ex = (int32_t)P.w; if (ey > (int32_t)P.h) ey = (int32_t)P.h;
        sc = (VkRect2D){{sx, sy}, {ex > sx ? (uint32_t)(ex - sx) : 0, ey > sy ? (uint32_t)(ey - sy) : 0}};
    }
    if (!sk || !g_ds_last.has_sc || memcmp(&g_ds_last.sc, &sc, sizeof sc)) { vkCmdSetScissor(cb, 0, 1, &sc); g_ds_last.sc = sc; g_ds_last.has_sc = 1; } else g_state_skips++;
    float db[3] = {S.po.units, S.po.clamp, S.po.factor};
    if (!sk || !g_ds_last.has_db || memcmp(g_ds_last.db, db, sizeof db)) { vkCmdSetDepthBias(cb, S.po.units, S.po.clamp, S.po.factor); memcpy(g_ds_last.db, db, sizeof db); g_ds_last.has_db = 1; } else g_state_skips++;
    if (!sk || !g_ds_last.has_bc || memcmp(g_ds_last.bc, S.blend_color, sizeof g_ds_last.bc)) { vkCmdSetBlendConstants(cb, S.blend_color); memcpy(g_ds_last.bc, S.blend_color, sizeof g_ds_last.bc); g_ds_last.has_bc = 1; } else g_state_skips++;
    for (int f = 0; f < 2; f++) {
        VkStencilFaceFlags ff = f ? VK_STENCIL_FACE_BACK_BIT : VK_STENCIL_FACE_FRONT_BIT;
        uint32_t st3[3] = {S.st_vmask[f], S.st_wmask[f], S.st_ref[f]};
        if (!sk || !g_ds_last.has_st[f] || memcmp(g_ds_last.st[f], st3, sizeof st3)) {
            vkCmdSetStencilCompareMask(cb, ff, S.st_vmask[f]); vkCmdSetStencilWriteMask(cb, ff, S.st_wmask[f]); vkCmdSetStencilReference(cb, ff, S.st_ref[f]);
            memcpy(g_ds_last.st[f], st3, sizeof st3); g_ds_last.has_st[f] = 1;
        } else g_state_skips++;
    }
    // vertex streams: only the vertices this draw reads are copied (games bind whole vertex pools, tens of MB). Per-vertex streams are copied from vertex vlo on and the draw's
    // vertex numbers shifted down by vlo (a shader reading its vertex index gets vlo added back: spv_vid_bias); per-instance streams from instance 0 up to the last one drawn
    uint32_t inst = dw ? (dw->instances ? (uint32_t)dw->instances : 1) : (di->instances ? (uint32_t)di->instances : 1);
    uint32_t isz = di ? (di->itype == 0 ? 1 : di->itype == 1 ? 2 : 4) : 0;
    const uint8_t *isrc = di ? (const uint8_t *)(uintptr_t)di->ibuf : NULL;
    #define IDX(i) (isz == 1 ? isrc[i] : isz == 2 ? ((const uint16_t *)isrc)[i] : ((const uint32_t *)isrc)[i])
    int64_t vlo = 0, vhi = -1;
    if (ind_b) { vlo = 0; vhi = INT32_MAX; }   // the GPU's own arguments: every vertex of the streams
    else if (dw) { vlo = dw->first; vhi = (int64_t)dw->first + dw->count - 1; }
    else if (di->count > 0) {
        // The vertices an index range uses: scanned once, then cached by its address, count and type for this generation while no write lands on it (the depth pass and the base
        // pass draw the same meshes; the scan was most of draw's own time)
        static struct { uint64_t a; int32_t n; uint8_t isz; uint64_t gen; uint32_t seq, lo, hi; } ic[2048];
        typeof(ic[0]) *e = &ic[(di->ibuf >> 4 ^ (uint64_t)di->count * 31) % 2048]; uint64_t ia = di->ibuf, ihi = ia + (uint64_t)di->count * isz;
        uint32_t lo = UINT32_MAX, hi = 0; int hit = 0;
        if (e->a == ia && e->n == di->count && e->isz == isz && e->gen == g_wgen && e->seq <= g_nwlog) {
            uint32_t k = e->seq; for (; k < g_nwlog; k++) if (g_wlo[k] < ihi && g_whi[k] > ia) break;
            g_h0_wscan[1] += k - e->seq; g_h0_wscan_n[1]++;
            if (k == g_nwlog) { e->seq = g_nwlog; lo = e->lo; hi = e->hi; hit = 1; }
        }
        if (!hit) {
            idx_scan(isrc, di->count, isz, &lo, &hi);   // scalar, or NEON with DM_VK_IDX_NEON (vk19.60)
            g_ic_scan++; g_ic_idx += (uint64_t)di->count; ic_key(ia, di->count, isz);
            if (e->a == ia && e->n == di->count && e->isz == isz) { g_ic_same++; g_h0_ic_rescan[e->gen >= g_h0_sub_gen0]++; } else if (e->a && e->gen == g_wgen) g_ic_coll++; else g_ic_new++;   // the same key again (earlier submission or written since) / a live key pushed out / a free or old slot
            shadow_note(1, ia, (uint64_t)di->count * isz);
            *e = (typeof(ic[0])){ia, di->count, (uint8_t)isz, g_wgen, g_nwlog, lo, hi};
        } else g_ic_hit++;
        if (lo <= hi) { vlo = (int64_t)lo + di->base_vertex; vhi = (int64_t)hi + di->base_vertex; }
    }
    if (vhi < vlo) { g_stat_draws++; return; }
    if (vlo < 0) vlo = 0;
    if (vs->vid) { int32_t bias = (int32_t)vlo; vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_VERTEX_BIT, 0, 4, &bias); g_vid_push++; if (bias) g_vid_nz++; }   // the shader adds it back to VertexIndex (spv_vid_bias)
    if ((dw ? dw->base_instance : di->base_instance) && !ind_b) g_h0_dbi++;   // H0
    if (g_h0_cpass && h0_on()) { g_h0_cp_draws++; if (!ind_b) g_h0_cp_prims += (uint64_t)(dw ? (dw->count > 0 ? dw->count : 0) : (di->count > 0 ? di->count : 0)) * inst;
                      for (int s2 = 0; s2 < 16; s2++) if (S.vbuf[s2].address) g_h0_cp_vbytes += S.vbuf[s2].size; }   // H0: the composed passes' draws
    uint32_t used = 0, ext[16] = {0};
    for (int i = 0; i < S.nattr; i++) if (vs->vin_mask >> i & 1) { uint32_t st = S.attr[i].stream, e = S.attr[i].offset + 16; used |= 1u << st; if (e > ext[st]) ext[st] = e; }
    VkBuffer vbs[16]; VkDeviceSize vos[16];
    for (int s = 0; s < 16; s++) if (used >> s & 1) {
        VkDeviceSize o = 0; VkBuffer b = g_zero_buf;
        uint64_t stride = s < S.nstream ? S.stream[s].stride : 0, size = S.vbuf[s].size, from, to;
        int per_inst = s < S.nstream && S.stream[s].divisor;
        if (ind_b) { from = 0; to = size < (32u << 20) ? size : (32u << 20); }
        else if (per_inst) { from = 0; to = ((uint64_t)(dw ? dw->base_instance : di->base_instance) + inst - 1) * stride + ext[s]; }
        else { from = (uint64_t)vlo * stride; to = (uint64_t)vhi * stride + ext[s]; }
        if (to > size) to = size;
        uint64_t h0c = g_h0_cmp[0][H0C_VTX] + g_h0_cmp[1][H0C_VTX]; g_h0_caller = H0C_VTX;
        if (S.vbuf[s].address && to > from && gw_find(S.vbuf[s].address + from, to - from, 1, &b, &o, NULL)) {   // vertices a dispatch wrote (GPU particles)
            static int h0v; if (h0v < 16 && g_h0_cmp[0][H0C_VTX] + g_h0_cmp[1][H0C_VTX] != h0c) { h0v++; char ab[256]; int ao = 0;   // H0: a composed vertex stream, in full
                for (int i = 0; i < S.nattr && ao < (int)sizeof ab - 32; i++) if ((vs->vin_mask >> i & 1) && S.attr[i].stream == (uint32_t)s) ao += snprintf(ab + ao, sizeof ab - ao, " a%d@%u fmt %#x vk %d", i, (unsigned)S.attr[i].offset, (unsigned)S.attr[i].fmt, (int)vfmt(S.attr[i].fmt));
                LOG("vk: H0 composed vertex stream %d: stride %llu, bound %llu bytes, read [%llu, %llu) (vertices %lld..%lld, extent %u, %s draw);%s", s, (unsigned long long)stride, (unsigned long long)size,
                    (unsigned long long)from, (unsigned long long)to, (long long)vlo, (long long)vhi, ext[s], ind_b ? "an indirect" : per_inst ? "a per-instance" : "a", ab); } }
        else if (S.vbuf[s].address && to > from && zc_mode() && zc_find(S.vbuf[s].address + from, to - from, &b, &o)) {}   // in a zero-copy pool: read where it is
        else if (S.vbuf[s].address && to > from && tog(TG_BSTORE) && ts_buf(S.vbuf[s].address + from, to - from, 16, TSK_VTX, &b, &o)) {}   // DM_VK_BUF_STORE (vk19.72): the store's block
        else if (S.vbuf[s].address && to > from) { VkDeviceSize ro = copy_in(S.vbuf[s].address + from, to - from, 16, CI_VTX); if (ro != VK_WHOLE_SIZE) { b = C->ring; o = ro; } }
        vbs[s] = b; vos[s] = o;
        if (h0slot >= 0 && !per_inst && stride && to > from) { uint64_t vc = (to - from) / stride; if (vc < C->h0vmax[h0slot]) C->h0vmax[h0slot] = (uint32_t)(vc > UINT32_MAX ? UINT32_MAX : vc); }   // H0: vertices copied
    }
    if (zero_attr_on()) { int z = 0; for (int i = 0; i < 16 && !z; i++) z = (vs->vin_mask >> i & 1) && (i >= S.nattr || !S.attr[i].fmt);
        if (z) { VkDeviceSize zo = 0; vkCmdBindVertexBuffers(cb, ZB, 1, &g_zero_buf, &zo); g_zattr_draws++; } }   // the zero binding of format-0 attributes (pso_create)
    // one call for each run of consecutive slots that changed (DM_VK_STATE_SKIP): was a call per stream for every draw
    for (int s = 0; s < 16;) {
        if (!(used >> s & 1) || (sk && (g_ds_last.vb_ok >> s & 1) && g_ds_last.vb[s] == vbs[s] && g_ds_last.vo[s] == vos[s])) { if (used >> s & 1) g_bind_skips++; s++; continue; }
        int e = s; while (e < 16 && (used >> e & 1) && !(sk && (g_ds_last.vb_ok >> e & 1) && g_ds_last.vb[e] == vbs[e] && g_ds_last.vo[e] == vos[e])) e++;
        vkCmdBindVertexBuffers(cb, (uint32_t)s, (uint32_t)(e - s), vbs + s, vos + s);
        for (int k = s; k < e; k++) { g_ds_last.vb[k] = vbs[k]; g_ds_last.vo[k] = vos[k]; g_ds_last.vb_ok |= 1u << k; }
        s = e;
    }
    if (ind_b) {   // arguments the GPU writes (ind_args): the records are in the ring where the dispatch leaves them
        if (dw) DRAWQ(vkCmdDrawIndirect(cb, ind_b, ind_o, 1, 16));
        else {
            uint64_t n = pool_left(di->ibuf, 8u << 20); VkBuffer ibs = C->ring; VkDeviceSize o = VK_WHOLE_SIZE;
            if (h0_on()) h0_ipd(di->ibuf, n);   // H0: index starts and pools of these draws
            if (n && tog(TG_BSTORE) && ts_buf(di->ibuf, n, 16, TSK_IDXPOOL, &ibs, &o)) {}   // DM_VK_BUF_STORE (vk19.72): the store's block
            else if (n && idxpool_once_mode() && ipo_get(di->ibuf, n, isz, &ibs, &o)) {}   // DM_VK_IDXPOOL_ONCE: the submission's one copy of the pool
            else { o = n ? copy_in(di->ibuf, n, 16, CI_IDXPOOL) : VK_WHOLE_SIZE; ibs = C->ring; if (h0slot >= 0 && o != VK_WHOLE_SIZE) C->h0ipp[h0slot] = C->ringp + o; }
            if (o == VK_WHOLE_SIZE) { RED_ONCE(38, "a GPU-written indirect indexed draw: its index buffer is not in a known pool"); return; }
            if (h0slot >= 0) { C->h0iw[h0slot] = (uint32_t)n; C->h0isz[h0slot] = (uint8_t)isz; }
            g_ds_last.ib_ok = 0; vkCmdBindIndexBuffer(cb, ibs, o, isz == 1 ? VK_INDEX_TYPE_UINT8_EXT : isz == 2 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32);
            DRAWQ(vkCmdDrawIndexedIndirect(cb, ind_b, ind_o, 1, 20));
        }
    } else if (dw) {
        if (prim == NVN_DRAW_PRIMITIVE_QUADS) {   // quads as two triangles each, through indices made here
            uint32_t nq = (uint32_t)dw->count / 4; if (!nq) return;
            VkDeviceSize o = ring_alloc((VkDeviceSize)nq * 24, 16); if (o == VK_WHOLE_SIZE) return;
            uint32_t *ix = (uint32_t *)(C->ringp + o);
            for (uint32_t q = 0; q < nq; q++) { uint32_t b = q * 4; uint32_t v[6] = {b, b + 1, b + 2, b, b + 2, b + 3}; memcpy(ix + 6 * q, v, 24); }
            g_ds_last.ib_ok = 0; vkCmdBindIndexBuffer(cb, C->ring, o, VK_INDEX_TYPE_UINT32);
            DRAWQ(vkCmdDrawIndexed(cb, nq * 6, inst, 0, (int32_t)(dw->first - vlo), (uint32_t)dw->base_instance));
        } else DRAWQ(vkCmdDraw(cb, (uint32_t)dw->count, inst, (uint32_t)(dw->first - vlo), (uint32_t)dw->base_instance));
    } else {
        int32_t voff = (int32_t)(di->base_vertex - vlo);
        if (prim == NVN_DRAW_PRIMITIVE_QUADS || (isz == 1 && !g_has_u8)) {   // quads, or 8-bit indices without the extension: widened to 32 bits on the CPU
            uint32_t n = (uint32_t)di->count, nq = prim == NVN_DRAW_PRIMITIVE_QUADS ? n / 4 : 0, out = nq ? nq * 6 : n;
            VkDeviceSize o = ring_alloc((VkDeviceSize)out * 4, 16); if (o == VK_WHOLE_SIZE) return;
            uint32_t *ix = (uint32_t *)(C->ringp + o);
            if (nq) for (uint32_t q = 0; q < nq; q++) { uint32_t b = 4 * q; uint32_t v[6] = {IDX(b), IDX(b + 1), IDX(b + 2), IDX(b), IDX(b + 2), IDX(b + 3)}; memcpy(ix + 6 * q, v, 24); }
            else for (uint32_t i = 0; i < n; i++) ix[i] = IDX(i);
            g_ds_last.ib_ok = 0; vkCmdBindIndexBuffer(cb, C->ring, o, VK_INDEX_TYPE_UINT32);
            DRAWQ(vkCmdDrawIndexed(cb, out, inst, 0, voff, (uint32_t)di->base_instance));
        } else {
            VkBuffer ibb = C->ring; VkDeviceSize o;
            if (!(zc_mode() && zc_find(di->ibuf, (uint64_t)di->count * isz, &ibb, &o) && o % isz == 0)) { ibb = C->ring; o = copy_in(di->ibuf, (uint64_t)di->count * isz, 16, CI_IDX); if (o == VK_WHOLE_SIZE) return; ibb = C->ring; }   // a zero-copy pool's indices where they are
            VkIndexType it = isz == 1 ? VK_INDEX_TYPE_UINT8_EXT : isz == 2 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
            if (sk && o % isz == 0 && o / isz <= UINT32_MAX) {   // the buffer bound once at offset 0, this draw's indices reached by firstIndex (DM_VK_STATE_SKIP)
                if (!g_ds_last.ib_ok || g_ds_last.ib != ibb || g_ds_last.it != (int)it) { vkCmdBindIndexBuffer(cb, ibb, 0, it); g_ds_last.ib = ibb; g_ds_last.it = (int)it; g_ds_last.ib_ok = 1; } else g_bind_skips++;
                DRAWQ(vkCmdDrawIndexed(cb, (uint32_t)di->count, inst, (uint32_t)(o / isz), voff, (uint32_t)di->base_instance));
            } else {
                vkCmdBindIndexBuffer(cb, ibb, o, it); g_ds_last.ib_ok = 0;
                DRAWQ(vkCmdDrawIndexed(cb, (uint32_t)di->count, inst, 0, voff, (uint32_t)di->base_instance));
            }
        }
    }
    #undef IDX
    trace_add('d', vs->key, fs ? fs->key : NULL, ind_b ? 0 : (uint32_t)(dw ? dw->count : di->count), inst, 0, ind_b != VK_NULL_HANDLE, g_ws_zero_tb);
    g_stat_draws++;
}
// Compute: the bound compute program's shader, its own pipeline and layout (one set), recorded outside render passes; afterwards every storage buffer it had is
// GPU-written: written back to guest memory at completion and read from the ring by the work recorded until then (gw_add)
static void dispatch(const p_dispatch *d) {
    if (vk_skip() & SK_DRAW) return;
    const prog_t *p = S.cprog; if (!p) return;
    // Indirect: the three group counts read here when the CPU can (ind_args), else by the GPU from the ring where an earlier dispatch of this submission writes them
    uint32_t g[3] = {(uint32_t)d->x, (uint32_t)d->y, (uint32_t)d->z}; VkBuffer ib = VK_NULL_HANDLE; VkDeviceSize io = 0;
    if (d->ind) { if (ind_args(d->ind, 12, g, &ib, &io, 1) == 1) ib = VK_NULL_HANDLE;
                  else { static int once; if (!once++) LOG("vk: an indirect dispatch's group counts come from a dispatch of the same submission: read by the GPU (logged once)"); } }
    else if (d->x <= 0 || d->y <= 0 || d->z <= 0) return;
    if (!ib) {
        if (!g[0] || !g[1] || !g[2]) return;
        for (int i = 0; i < 3; i++) if (g[i] > g_props.limits.maxComputeWorkGroupCount[i]) {   // past the GPU's limit: undefined, and a lost device on some
            static int n; if (n++ < 4) LOG("[RED] vk: dispatch of %ux%ux%u groups exceeds the GPU's %ux%ux%u: skipped", g[0], g[1], g[2], g_props.limits.maxComputeWorkGroupCount[0], g_props.limits.maxComputeWorkGroupCount[1], g_props.limits.maxComputeWorkGroupCount[2]); return; }
    }
    static struct { const prog_t *p; uint64_t ser; shader_t *cs; } pc[256];
    unsigned h = (unsigned)(((uintptr_t)p >> 4) ^ p->serial) % 256;
    if (pc[h].p != p || pc[h].ser != p->serial) {
        pc[h].p = p; pc[h].ser = p->serial; pc[h].cs = NULL;
        for (unsigned i = 0; i < p->nstage; i++) { uint8_t md[8]; if (!prog_key(p, i, md)) continue; shader_t *sh = shader_get(md); if (sh->stage == 5) pc[h].cs = sh; }
    }
    shader_t *cs = pc[h].cs; if (!cs || !cs->mod) { RED_ONCE(64, "compute program without a usable compute shader: dispatch skipped"); return; }
    if (!cs->cpp) {
        VkPipelineLayoutCreateInfo li = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &cs->dsl, 0, NULL}; VKCHECK(vkCreatePipelineLayout(g_dev, &li, NULL, &cs->cpl));
        VkComputePipelineCreateInfo ci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0, {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, cs->mod, "main", NULL}, cs->cpl, VK_NULL_HANDLE, -1};
        if (vkCreateComputePipelines(g_dev, g_pcache, 1, &ci, NULL, &cs->cpp) != VK_SUCCESS) { LOG("[RED] vk: compute pipeline for %s not created", cs->key); cs->mod = VK_NULL_HANDLE; return; }
    }
    int dfr = defer_on() && P.open && !ib && g_ndefer < NDEFER;   // recorded after the open pass instead of ending it (defer_on)
    if (!dfr) pass_end();
    g_ws_zero_tb = 0; g_ws_att = 0; g_nws_simg = 0;
    VkDescriptorSet ds = write_set(cs, 5); if (!ds) return;
    if (dfr && (!P.open || g_ws_att)) { dfr = 0; pass_end(); }   // it reads one of the pass's attachments (or the pass ended meanwhile): as before
    if (dfr) {
        dfr_t *q = &g_dfr[g_ndefer++]; q->cs = cs; q->ds = ds; q->ndyno = g_ndyno < 64 ? g_ndyno : 64; memcpy(q->dyno, g_dyno, q->ndyno * 4); q->g[0] = g[0]; q->g[1] = g[1]; q->g[2] = g[2]; q->zero_tb = g_ws_zero_tb;
        for (int i = 0; i < g_nws_simg; i++) g_ws_simg[i]->dfr_gen = g_defer_gen;
        g_ndefer_n++;
        if (g_plog) LOG("plog:  dispatch cs %s %ux%ux%u deferred past the open pass, %d storage buffers", cs->key, g[0], g[1], g[2], g_nwb);
    } else {
    if (g_plog) LOG("plog:  dispatch cs %s %ux%ux%u%s, %d storage buffers", cs->key, g[0], g[1], g[2], ib ? " (indirect, GPU-written)" : "", g_nwb);
    vkCmdBindPipeline(C->cb, VK_PIPELINE_BIND_POINT_COMPUTE, cs->cpp);
    vkCmdBindDescriptorSets(C->cb, VK_PIPELINE_BIND_POINT_COMPUTE, cs->cpl, 0, 1, &ds, g_ndyno, g_dyno);
    { char b[64]; snprintf(b, sizeof b, "cs %s", cs->key); g_tsig_next = fnv(cs->key, strlen(cs->key)) | 1; gsig_name(g_tsig_next, b); }
    gpu_mark(1, 0);
    if (ib) vkCmdDispatchIndirect(C->cb, ib, io); else vkCmdDispatch(C->cb, g[0], g[1], g[2]);
    gpu_mark(1, 1);
    trace_add('c', cs->key, NULL, ib ? 0 : g[0], ib ? 0 : g[1], ib ? 0 : g[2], ib != VK_NULL_HANDLE, g_ws_zero_tb);
    barrier_cs(C->cb);
    }
    g_gw_add_dfr = dfr;   // its written ranges: a draw reading one records the deferred dispatches first
    for (int i = 0; i < g_nwb; i++) {
        gw_t *w = &g_wb[i]; int from_gw = w->fence != 0;
        // Back into guest memory at completion, only what the GPU changed. The GPU writes the bytes a shader stores and no others: the whole range written back put the bytes the
        // dispatch was recorded with over everything the CPU wrote there while the submission ran (a frame or two in flight). So the bytes it starts from are kept (the guest's,
        // copied into the ring now), and the completion writes back the 4-byte words that differ from them (post kind 5). A range already set to go back in this submission keeps
        // its first entry (its snapshot holds the guest's bytes; the ring holds the final ones). A range another submission's GPU work wrote goes back whole, as before
        int dup = 0; const uint8_t *dsnap = NULL;
        for (int k = 0; k < C->npost && !dup; k++) { const post_t *q = &C->post[k]; uint64_t qa = (uint64_t)(uintptr_t)q->dst;
            if (q->kind == 3 && qa == w->addr && q->src == w->p && q->n == w->size) dup = 1;   // the same range again
            else if (q->kind == 5 && w->addr >= qa && w->addr + w->size <= qa + q->n && q->src + (w->addr - qa) == w->p) { dup = 1; dsnap = (const uint8_t *)(uintptr_t)q->addr + (w->addr - qa); } }   // inside one that goes back by its changed words
        uint8_t *snap = !dup && !from_gw && w->ci == g_ci && w->p ? snap_alloc(w->size) : NULL;
        if (from_gw) { g_h0_chain_n++; g_h0_chain_b += w->size; } else g_h0_dsp_fresh++;   // H0
        if (snap) memcpy(snap, w->p, w->size);
        int chained = 0;
        if (!snap && !dup && from_gw && w->p && chain_snap_mode() && (snap = snap_alloc(w->size))) {   // DM_VK_GW_CHAIN_SNAP: the guest's bytes now (the GPU starts from the running range's)
            memcpy(snap, (const void *)(uintptr_t)w->addr, w->size); chained = 1; g_cs_n++; g_cs_b += w->size; }
        // The range is the GPU's until the submission completes (gw_find), with the bytes it started from to tell the CPU's later rewrites (gw_cpu_dirty)
        gw_add(w->addr, w->size, w->buf, w->off, w->p, g_fence + 1, w->ci, snap ? snap : dsnap, 0);   // this submission's fence value (mtl_sub_commit assigns g_fence + 1)
        if (dup) continue;
        if (snap) post_add((post_t){5, (void *)(uintptr_t)w->addr, (uint32_t)chained, 0, (uint64_t)(uintptr_t)snap, 0, w->p, w->size});   // mode 1: a chained range
        else {   // H0: a chained range (no snapshot) goes back whole; the guest's bytes now are kept aside (measurement only) to count what the CPU writes meanwhile that the write-back overwrites
            uint8_t *h0s = NULL; if (from_gw && h0_on()) { if (w->size <= (1u << 20)) { h0s = malloc(w->size); if (h0s) memcpy(h0s, (const void *)(uintptr_t)w->addr, w->size); } else atomic_fetch_add(&g_h0_snap_skip, 1); }
            post_add((post_t){3, (void *)(uintptr_t)w->addr, 0, 0, (uint64_t)(uintptr_t)h0s, 0, w->p, w->size}); }
    }
    g_gw_add_dfr = 0;
    if (gw_shadow_on() && !dfr) for (int i = 0; i < g_nwb; i++) {   // DM_VK_GW_SHADOW: the remembered misaligned reads of these ranges, copied aligned now (after barrier_cs, no pass open)
        gw_t *w = &g_wb[i]; uint64_t ro, rs; if (!w->buf || !shl_find(w->addr, w->size, &ro, &rs)) continue;
        VkDeviceSize al = g_props.limits.minStorageBufferOffsetAlignment > 256 ? g_props.limits.minStorageBufferOffsetAlignment : 256;
        VkDeviceSize o = ring_alloc(rs, al); if (o == VK_WHOLE_SIZE) continue;
        VkBufferCopy bc = {w->off + ro, o, rs}; vkCmdCopyBuffer(C->cb, w->buf, C->ring, 1, &bc); barrier_xfer(C->cb);
        gw_add(w->addr + ro, rs, C->ring, o, C->ringp + o, g_fence + 1, g_ci, NULL, 1); g_shl_made++;
    }
    g_stat_draws++;
}
static void clear_color(const p_clear_color *c) {
    if (c->index < 0 || c->index >= P.n || !P.cv[c->index]) return;
    if ((c->mask & 15) != 15) RED_ONCE(40, "a color clear with a partial channel mask clears every channel");
    if (!P.open) { P.clr_c[c->index] = 1; memcpy(&P.cc[c->index], c->rgba, 16); return; }   // before the pass's first draw: its load action
    VkClearAttachment a = {VK_IMAGE_ASPECT_COLOR_BIT, (uint32_t)c->index, {.color = {{c->rgba[0], c->rgba[1], c->rgba[2], c->rgba[3]}}}};
    VkClearRect r = {{{0, 0}, {P.w, P.h}}, 0, P.layered ? P.nlayers : 1};   // a layered pass clears every layer, as NVN clears every layer of a layered target
    vkCmdClearAttachments(C->cb, 1, &a, 1, &r); P.pw_c[c->index] = 1;
}
static void clear_ds(const p_clear_ds *c) {
    if (!P.dt) return;
    if (!P.open) { if (c->depth_mask) { P.clr_d = 1; P.cd = c->depth; } if (P.dt->f.stencil && c->stencil_mask) { P.clr_s = 1; P.cs = (uint32_t)c->stencil; } return; }
    if (!pass_begin()) return;
    VkImageAspectFlags asp = (c->depth_mask ? VK_IMAGE_ASPECT_DEPTH_BIT : 0) | (P.dt->f.stencil && c->stencil_mask ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
    if (!asp) return;
    VkClearAttachment a = {asp, 0, {.depthStencil = {c->depth, (uint32_t)c->stencil}}};
    VkClearRect r = {{{0, 0}, {P.w, P.h}}, 0, P.layered ? P.nlayers : 1};   // a layered pass clears every layer, as NVN clears every layer of a layered target
    vkCmdClearAttachments(C->cb, 1, &a, 1, &r); P.pw_d = 1;
}
// Copies and blits are checked against the image before they are recorded: a region past a level's extent, a level or layer the image lacks, or a blit the format does not support
// (VK_FORMAT_FEATURE_BLIT_SRC / DST, linear filtering) is invalid usage, and the Mali-G57 answered such work in Super Mario Odyssey with a lost device. Equal-size blits fall back to a copy
static int lvl_fits(const vtex *v, uint32_t lv, uint32_t layer, uint32_t nlayers, int32_t x, int32_t y, int32_t z, uint32_t w, uint32_t h, uint32_t d) {
    if (lv >= v->levels || x < 0 || y < 0 || z < 0) return 0;
    uint32_t lw = v->w >> lv ? v->w >> lv : 1, lh = v->h >> lv ? v->h >> lv : 1, ld = v->d >> lv ? v->d >> lv : 1;
    if (v->target == NVN_TEXTURE_TARGET_1D_ARRAY) lh = 1;
    if ((uint64_t)x + w > lw || (uint64_t)y + h > lh || (uint64_t)z + d > ld) return 0;
    return v->target == NVN_TEXTURE_TARGET_3D || (uint64_t)layer + nlayers <= v->layers;
}
static VkFormatFeatureFlags fmt_feat(VkFormat f) { VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties(g_pd, f, &fp); return fp.optimalTilingFeatures; }
static int copylog_on(void) {   // DM_VK_COPYLOG=<n>: log the first n buffer-to-texture and texture copies
    static int left = -1; if (left < 0) { const char *e = getenv("DM_VK_COPYLOG"); left = e ? atoi(e) : 0; if (left) LOG("DM_VK_COPYLOG=%d: copies into textures are logged", left); }
    if (left <= 0) return 0;
    left--; return 1;
}
// A GPU copy's bytes also go into the destination's storage when the texture takes its content from there (as nvn_mtl.m tex_storage_dst / nvn_tex_mirror_*): any later upload
// from the storage -- the texture made again after the game released it, a new texture object over the same storage, an eviction -- then has the copied levels too. The level
// stays the GPU's for this image (gpu_lv); the storage now holds the same bytes (mir_lv)
int nvn_tex_mirror_b2t(const tex_t *t, unsigned level, unsigned layer0, const NVNcopyRegion *r, const uint8_t *src, size_t row, size_t img);   // hle_nvn.c
int nvn_tex_mirror_t2t(const tex_t *s, unsigned sl, unsigned ss, const NVNcopyRegion *sr, const tex_t *d, unsigned dl, unsigned ds, const NVNcopyRegion *dr);
static uint64_t g_mir_b2t, g_mir_t2t;
static int mirror_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_MIRROR"); v = !(e && !strcmp(e, "0")); if (!v) LOG("DM_VK_MIRROR=0: GPU copies into textures are not written into their storage"); } return v; }   // DM_VK_MIRROR=0: off
static int storage_backed(const tex_t *t, const vtex *v) {
    if (!mirror_on()) return 0;
    if (!t->mem || !t->pool || (t->pool->flags & (NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS | NVN_MEMORY_POOL_FLAGS_VIRTUAL)) || t->d.samples > 1) return 0;
    if (t->d.target == NVN_TEXTURE_TARGET_BUFFER || t->d.target == NVN_TEXTURE_TARGET_3D || (t->d.flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET | NVN_TEXTURE_FLAGS_COMPRESSIBLE | NVN_TEXTURE_FLAGS_IMAGE))) return 0;
    return v && !v->gpu;
}
static int storage_holds(const vtex *v, unsigned lv) { return v && !v->gpu && lv < 32 && ((v->packaged && !(v->gpu_lv >> lv & 1)) || (v->mir_lv >> lv & 1)); }   // the level's bytes are in the storage
static void copy_b2t(const p_copy_b2t *c) {
    if (vk_skip() & SK_B2T) return;
    if (g_plog) LOG("plog:  b2t serial %llu fmt %#x %dx%d at %d,%d,%d", (unsigned long long)c->tex->serial, c->tex->d.format, c->r.width, c->r.height, c->r.xoffset, c->r.yoffset, c->r.zoffset);
    vtex *v = tex_get(c->tex, 0); if (!v) return;
    if (v->f.dec) { RED_ONCE(50, "buffer-to-texture copy into a BC texture: not decoded yet"); return; }
    v->copied = 1;
    int bw = 1, bh = 1, bpb = nvn_fmt_bpb(c->tex->d.format, &bw, &bh); if (!bpb) return;
    const NVNcopyRegion *r = &c->r; uint32_t w = (uint32_t)r->width, h = (uint32_t)r->height, d = (uint32_t)(r->depth > 0 ? r->depth : 1);
    size_t row = S.copy_stride ? (size_t)S.copy_stride : (size_t)((w + bw - 1) / bw) * bpb, img = S.copy_img_stride ? (size_t)S.copy_img_stride : row * ((h + bh - 1) / bh);
    pass_end();
    // The source as the GPU will hold it: what a dispatch or a texture-to-buffer copy wrote there (this submission's, or an earlier one's still running) is only in a ring until
    // its submission completes, and the guest's bytes are stale until then; else the guest's bytes
    g_h0_caller = H0C_IMG; VkBuffer sb; VkDeviceSize o; uint8_t *sp = NULL; int from_gpu = gw_exact(c->src, img * d) && gw_find(c->src, img * d, 16, &sb, &o, &sp);
    if (!from_gpu) { o = copy_in(c->src, img * d, 16, CI_IMG); if (o == VK_WHOLE_SIZE) return; sb = C->ring; sp = C->ringp + o; }   // the ring after copy_in: a full one is replaced by an extra ring
    else atomic_fetch_add(&g_gw_b2t, 1);
    if (copylog_on()) {   // diagnostics (DM_VK_COPYLOG=<n>): the first n copies into textures, with what the copy reads, to set beside the Metal host's [pass] copy lines
        uint64_t hh = 0xcbf29ce484222325ull, nz = 0, nb = img * d; if (sp) for (uint64_t i = 0; i < nb; i++) { hh = (hh ^ sp[i]) * 0x100000001b3ull; nz += sp[i] != 0; }
        LOG("copylog: frame %llu b2t serial %llu fmt %#x %ux%ux%u at %d,%d,%d from %#llx row %zu image %zu, %s, %llu bytes, %llu non-zero, hash %016llx", (unsigned long long)atomic_load(&g_presents),
            (unsigned long long)c->tex->serial, c->tex->d.format, w, h, d, r->xoffset, r->yoffset, r->zoffset, (unsigned long long)c->src, row, img, from_gpu ? "GPU-written" : "guest memory",
            (unsigned long long)nb, (unsigned long long)nz, (unsigned long long)hh); }
    uint32_t lv = c->has_view && c->view.set_levels ? c->view.base_level : 0, ly = c->has_view && c->view.set_layers ? c->view.min_layer : 0;
    if (lv < 32) v->gpu_lv |= 1u << lv;   // this level's content now comes from copies (see copy_t2t)
    if (!from_gpu && storage_backed(c->tex, v) && !nvn_tex_mirror_b2t(c->tex, lv, ly, r, (const uint8_t *)(uintptr_t)c->src, row, img)) { if (lv < 32) v->mir_lv |= 1u << lv; g_mir_b2t++; }
    int is3d = v->target == NVN_TEXTURE_TARGET_3D;
    if (!lvl_fits(v, lv, is3d ? 0 : ly + (uint32_t)r->zoffset, is3d ? 1 : d, r->xoffset, r->yoffset, is3d ? r->zoffset : 0, w, h, is3d ? d : 1)) {
        static int n; if (n++ < 8) LOG("[RED] vk: buffer-to-texture copy %dx%dx%d at %d,%d,%d level %u outside the %ux%ux%u texture (%u levels, %u layers): skipped", (int)w, (int)h, (int)d, r->xoffset, r->yoffset, r->zoffset, lv, v->w, v->h, v->d, v->levels, v->layers);
        return;
    }
    VkBufferImageCopy bc = {o, (uint32_t)(row / bpb * bw), (uint32_t)(img / row * bh), {v->aspect & VK_IMAGE_ASPECT_COLOR_BIT ? VK_IMAGE_ASPECT_COLOR_BIT : VK_IMAGE_ASPECT_DEPTH_BIT, lv, is3d ? 0 : ly + (uint32_t)r->zoffset, is3d ? 1 : d},
                            {r->xoffset, r->yoffset, is3d ? r->zoffset : 0}, {w, h, is3d ? d : 1}};
    if (v->ubwc) g_ubwc_b2t++;
    gpu_mark(2, 0); vkCmdCopyBufferToImage(C->cb, sb, v->img, VK_IMAGE_LAYOUT_GENERAL, 1, &bc); gpu_mark(2, 1);
    barrier_xfer(C->cb);
}
static void copy_t2t(const p_copy_t2t *c) {
    if (vk_skip() & SK_T2T) return;
    if (g_plog) LOG("plog:  t2t %llu (fmt %#x) %dx%d → %llu (fmt %#x) %dx%d", (unsigned long long)c->src->serial, c->src->d.format, c->sr.width, c->sr.height, (unsigned long long)c->dst->serial, c->dst->d.format, c->dr.width, c->dr.height);
    // The destination is fetched as a texture, not a render target: Unreal's texture streaming copies the mips two textures share on the GPU and streams the others into the
    // new texture's storage; made as a render target it was never uploaded (and one made already stopped following its storage), so the streamed mips never arrived and sprites
    // stayed on their coarsest level, magnified into blocks (FINAL FANTASY RESONANCE's battle). The level copied into is the GPU's from now on (gpu_lv)
    vtex *s = tex_get(c->src, 0), *d = tex_get(c->dst, 0); if (!s || !d) return;
    pass_end();
    uint32_t sl = c->has_sv && c->sv.set_levels ? c->sv.base_level : 0, dl = c->has_dv && c->dv.set_levels ? c->dv.base_level : 0;
    if (dl < 32) d->gpu_lv |= 1u << dl;
    if (storage_backed(c->dst, d) && storage_holds(s, sl) && c->sr.width == c->dr.width && c->sr.height == c->dr.height
        && !nvn_tex_mirror_t2t(c->src, sl, (unsigned)c->sr.zoffset, &c->sr, c->dst, dl, (unsigned)c->dr.zoffset, &c->dr)) { if (dl < 32) d->mir_lv |= 1u << dl; g_mir_t2t++; }
    if (copylog_on()) LOG("copylog: frame %llu t2t serial %llu (fmt %#x, level %u, %s) %dx%d at %d,%d -> serial %llu (fmt %#x, level %u) at %d,%d", (unsigned long long)atomic_load(&g_presents),
                          (unsigned long long)c->src->serial, c->src->d.format, sl, s->gpu ? "rendered" : s->gpu_lv ? "copied into" : "from its storage", c->sr.width, c->sr.height, c->sr.xoffset, c->sr.yoffset,
                          (unsigned long long)c->dst->serial, c->dst->d.format, dl, c->dr.xoffset, c->dr.yoffset);
    VkImageAspectFlags sa = s->aspect & VK_IMAGE_ASPECT_COLOR_BIT ? VK_IMAGE_ASPECT_COLOR_BIT : VK_IMAGE_ASPECT_DEPTH_BIT, da = d->aspect & VK_IMAGE_ASPECT_COLOR_BIT ? VK_IMAGE_ASPECT_COLOR_BIT : VK_IMAGE_ASPECT_DEPTH_BIT;
    if (!lvl_fits(s, sl, (uint32_t)c->sr.zoffset, 1, c->sr.xoffset, c->sr.yoffset, 0, (uint32_t)c->sr.width, (uint32_t)c->sr.height, 1) ||
        !lvl_fits(d, dl, (uint32_t)c->dr.zoffset, 1, c->dr.xoffset, c->dr.yoffset, 0, (uint32_t)c->dr.width, (uint32_t)c->dr.height, 1) || !c->sr.width || !c->sr.height || !c->dr.width || !c->dr.height) {
        static int n; if (n++ < 8) LOG("[RED] vk: texture copy %dx%d at %d,%d (level %u) → %dx%d at %d,%d (level %u) outside its textures (%ux%u, %ux%u): skipped", c->sr.width, c->sr.height, c->sr.xoffset, c->sr.yoffset, sl,
                                       c->dr.width, c->dr.height, c->dr.xoffset, c->dr.yoffset, dl, s->w, s->h, d->w, d->h);
        return;
    }
    int same = c->sr.width == c->dr.width && c->sr.height == c->dr.height;
    VkFormatFeatureFlags sf = fmt_feat(s->f.f), df = fmt_feat(d->f.f);
    int lin = (c->flags & 1) && !same, blit_ok = (sf & VK_FORMAT_FEATURE_BLIT_SRC_BIT) && (df & VK_FORMAT_FEATURE_BLIT_DST_BIT) && (!lin || (sf & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) && sa == da;
    if (same && (!blit_ok || s->f.f == d->f.f) && sa == da) {   // one size: a copy (formats of one size class), which every format supports
        VkImageCopy ic = {{sa, sl, (uint32_t)c->sr.zoffset, 1}, {c->sr.xoffset, c->sr.yoffset, 0}, {da, dl, (uint32_t)c->dr.zoffset, 1}, {c->dr.xoffset, c->dr.yoffset, 0}, {(uint32_t)c->sr.width, (uint32_t)c->sr.height, 1}};
        if (d->ubwc) g_ubwc_t2t++;
        vkCmdCopyImage(C->cb, s->img, VK_IMAGE_LAYOUT_GENERAL, d->img, VK_IMAGE_LAYOUT_GENERAL, 1, &ic);
    } else if (blit_ok) {
        VkImageBlit b = {{sa, sl, (uint32_t)c->sr.zoffset, 1}, {{c->sr.xoffset, c->sr.yoffset, 0}, {c->sr.xoffset + c->sr.width, c->sr.yoffset + c->sr.height, 1}},
                         {da, dl, (uint32_t)c->dr.zoffset, 1}, {{c->dr.xoffset, c->dr.yoffset, 0}, {c->dr.xoffset + c->dr.width, c->dr.yoffset + c->dr.height, 1}}};
        if (d->ubwc) g_ubwc_t2t++;
        vkCmdBlitImage(C->cb, s->img, VK_IMAGE_LAYOUT_GENERAL, d->img, VK_IMAGE_LAYOUT_GENERAL, 1, &b, lin ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
    } else { static int n; if (n++ < 8) LOG("[RED] vk: scaled texture copy between formats %d → %d that cannot be blitted: skipped", (int)s->f.f, (int)d->f.f); return; }
    barrier_xfer(C->cb);
}
static void copy_t2b(const p_copy_t2t *c) {   // src = the texture, dst field of the record = the destination buffer address (see nvn_obj.h)
    if (vk_skip() & SK_T2B) return;
    if (g_plog) LOG("plog:  t2b %llu %dx%d", (unsigned long long)c->src->serial, c->sr.width, c->sr.height);
    vtex *s = tex_get(c->src, 0); if (!s) return;
    int bw = 1, bh = 1, bpb = nvn_fmt_bpb(c->src->d.format, &bw, &bh); if (!bpb || s->f.dec) { RED_ONCE(51, "texture-to-buffer copy of format %#x not implemented", c->src->d.format); return; }
    pass_end();
    uint32_t w = (uint32_t)c->sr.width, h = (uint32_t)c->sr.height; size_t row = S.copy_stride ? (size_t)S.copy_stride : (size_t)((w + bw - 1) / bw) * bpb, n = row * ((h + bh - 1) / bh);
    VkDeviceSize o = ring_alloc(n, 16); if (o == VK_WHOLE_SIZE) return;
    uint32_t sl = c->has_sv && c->sv.set_levels ? c->sv.base_level : 0;
    if (!lvl_fits(s, sl, (uint32_t)c->sr.zoffset, 1, c->sr.xoffset, c->sr.yoffset, 0, w, h, 1)) { static int nn; if (nn++ < 8) LOG("[RED] vk: texture-to-buffer copy %ux%u outside the %ux%u texture: skipped", w, h, s->w, s->h); return; }
    VkBufferImageCopy bc = {o, (uint32_t)(row / bpb * bw), 0, {s->aspect & VK_IMAGE_ASPECT_COLOR_BIT ? VK_IMAGE_ASPECT_COLOR_BIT : VK_IMAGE_ASPECT_DEPTH_BIT, sl, (uint32_t)c->sr.zoffset, 1},
                            {c->sr.xoffset, c->sr.yoffset, 0}, {w, h, 1}};
    gpu_mark(2, 0); vkCmdCopyImageToBuffer(C->cb, s->img, VK_IMAGE_LAYOUT_GENERAL, C->ring, 1, &bc); gpu_mark(2, 1);
    barrier_xfer(C->cb);
    // Until the submission completes and the bytes go back to guest memory, later reads of the destination (a buffer-to-texture copy, a buffer copy, a draw's buffers) take them
    // from the ring, as a dispatch's writes (gw_add); they read the guest's stale bytes before
    gw_add((uint64_t)(uintptr_t)c->dst, n, C->ring, o, C->ringp + o, g_fence + 1, g_ci, NULL, 1);   // an exact write: every byte of the range
    size_t packed = (size_t)((w + bw - 1) / bw) * bpb, rows = (h + bh - 1) / bh;
    if (row > packed) post_add((post_t){6, (void *)c->dst, (uint32_t)row, (uint32_t)rows, 0, (uint32_t)packed, C->ringp + o, n});   // a row stride past the texels: only the texels go back, the bytes between rows stay the guest's
    else post_add((post_t){3, (void *)c->dst, 0, 0, 0, 0, C->ringp + o, n});
}
static void clear_tex(const p_clear_tex *c) {
    if (vk_skip() & SK_CLEAR) return;
    if (g_plog) LOG("plog:  clear tex %llu fmt %#x", (unsigned long long)c->tex->serial, c->tex->d.format);
    vtex *v = tex_get(c->tex, 1); if (!v) return;
    pass_end();
    uint32_t lv = c->has_view && c->view.set_levels ? c->view.base_level : 0;
    if (lv >= v->levels) { static int n; if (n++ < 8) LOG("[RED] vk: clear of level %u of a %u-level texture: skipped", lv, v->levels); return; }
    VkClearColorValue cc; memcpy(&cc, &c->c, 16);
    VkImageSubresourceRange rr = {VK_IMAGE_ASPECT_COLOR_BIT, lv, 1, 0, VK_REMAINING_ARRAY_LAYERS};
    if (v->f.depth) { VkClearDepthStencilValue dv = {c->c.f[0], 0}; rr.aspectMask = v->aspect; vkCmdClearDepthStencilImage(C->cb, v->img, VK_IMAGE_LAYOUT_GENERAL, &dv, 1, &rr); }
    else vkCmdClearColorImage(C->cb, v->img, VK_IMAGE_LAYOUT_GENERAL, &cc, 1, &rr);
    barrier_xfer(C->cb);
}

// ---------- Submission ----------
// The context's own ring sized to what its submissions take (its previous one is complete, and nothing holds its ring: mtl_sub_begin waited for both)
static _Atomic uint64_t g_trim_gen, g_wait_fence_ns, g_wait_hold_ns;
void dm_vk_trim(void) { atomic_fetch_add(&g_trim_gen, 1); atomic_fetch_add(&g_ring_trim_gen, 1); }   // Android's own signal: real pressure
// The low-memory signal (onLowMemory) came too late: the low-memory killer ended the game in the foreground twice (FINAL FANTASY RESONANCE on the Xiaomi Pad 8, RSS 2.4 GB, the
// system swapping). Every second the backend reads MemAvailable and the process's resident memory; below DM_VK_MEM_LOW_MB (default 1500) available, or past DM_MEM_BUDGET_MB
// (default 2000) for the process's resident memory plus the backend's device memory, it trims as for the signal, at most every 5 s: idle textures evicted (60 frames unused), rings
// back to their minimum, the reuse pools emptied, malloc's free pages given back. The hibernation build's game heap gives its free chunks' pages back every 10 s, and at once then
#include <malloc.h>
size_t dm_ueheap_release(size_t min) __attribute__((weak));   // the hibernation build's game heap (hle_ueheap.c): free chunks' pages back to the system
static uint64_t g_heap_released;
// The game heap's release (dm_ueheap_release walks every space under its lock, 5% of the render thread in a profile when it ran here every 10 s) runs on a thread of its own,
// woken every 60 s and at once on low memory
static pthread_mutex_t g_hrel_mu = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t g_hrel_cv = PTHREAD_COND_INITIALIZER; static int g_hrel_req;
static void *heap_release_thread(void *a) { pthread_setname_np("dm-heaprel");
    (void)a; static size_t prev; static int nr;
    for (;;) {
        pthread_mutex_lock(&g_hrel_mu); while (!g_hrel_req) pthread_cond_wait(&g_hrel_cv, &g_hrel_mu); g_hrel_req = 0; pthread_mutex_unlock(&g_hrel_mu);
        size_t r = dm_ueheap_release(65536);
        if ((r > prev + (16u << 20) || r + (16u << 20) < prev) && nr++ < 40) LOG("vk: game heap: free blocks of 64 KB or more now %.0f MB, their pages given back to the system", r / 1048576.0);
        prev = r; g_heap_released = r;   // the total of the free blocks walked (pages already given back are counted again: madvise does not say)
    }
    return NULL;
}
static void heap_release_kick(void) {
    static int started; if (!started) { started = 1; pthread_t t; if (pthread_create(&t, NULL, heap_release_thread, NULL)) { started = 0; return; } pthread_detach(t); }
    pthread_mutex_lock(&g_hrel_mu); g_hrel_req = 1; pthread_cond_signal(&g_hrel_cv); pthread_mutex_unlock(&g_hrel_mu);
}
// Early trim (vk19.29, DM_MEM_PRETRIM, default on). The killed runs all had the game heap step from 0.75 to 1.05 GB, the process's resident memory 500 MB up within seconds, and the
// killer acted 4 s after MemAvailable began to fall: the trim at DM_VK_MEM_LOW_MB came too late to matter. Now the backend trims at once (at most every 2 s) when the game heap has grown
// DM_MEM_PRETRIM_MB (default 100) since the last one-second check, or MemAvailable fell twice that in a second; and as at the low mark (at most every 5 s) below
// DM_MEM_PRETRIM_AVAIL_MB (default 1600). After an early trim the backend keeps nothing for reuse for 10 s (g_mem_low). The heap's free pages go back on each trim and every 60 s
// (no longer every second while memory is low: the walk takes the heap's locks while the game allocates hardest). A line each second the heap, the resident memory or MemAvailable
// moved a lot ([mem] last second), at most 300
int64_t dm_ue_heap_used(void), dm_ue_heap_big(void);   // hle_uemalloc.c
// IO back-pressure (vk19.33): the memory watch samples MemAvailable every 100 ms; below DM_IO_THROTTLE_AVAIL_MB (default 900), or 300 MB lower than a second before, the file system's
// reads wait a little (hle_fs.c io_throttle) so that reclaim and swap catch up before the low-memory killer acts. dm_mem_io_pressure: MemAvailable in MB under pressure, else 0
static _Atomic long g_io_press;
int dm_mem_io_pressure(void) { return (int)atomic_load(&g_io_press); }
static void io_pressure_sample(uint64_t now) {
    static long hist[10]; static int hi, hn; static long lim = -1; static uint64_t last;
    if (now - last < 90000000ull) return; last = now;
    if (lim < 0) { const char *e = getenv("DM_IO_THROTTLE_AVAIL_MB"); lim = e ? atol(e) : 900; }
    long avail = 0; char ln[128]; FILE *f = fopen("/proc/meminfo", "re"); if (f) { while (fgets(ln, sizeof ln, f)) if (sscanf(ln, "MemAvailable: %ld", &avail) == 1) break; fclose(f); }
    if (avail <= 0) return;
    avail /= 1024; long old = hn == 10 ? hist[hi] : avail; hist[hi] = avail; hi = (hi + 1) % 10; if (hn < 10) hn++;
    atomic_store(&g_io_press, (avail < lim || old - avail > 300) ? (avail > 0 ? avail : 1) : 0);
}
static void mem_watch(void) {
    static uint64_t last, last_trim, last_rel, pre_until; static int64_t pbase = -1, hbase = -1, bbase; static long pavail, prss; static int nline;
    static long low = -1, budget = -1, pre_mb, pre_avail;
    if (low < 0) {
        const char *e = getenv("DM_VK_MEM_LOW_MB"); low = e ? atol(e) : 800; e = getenv("DM_MEM_BUDGET_MB"); budget = e ? atol(e) : 0;   // vk19.41 defaults (were 1500 / 2000): with the steady-low back-off
        e = getenv("DM_MEM_PRETRIM"); int on = !e || atoi(e) != 0;
        e = getenv("DM_MEM_PRETRIM_MB"); pre_mb = on ? (e ? atol(e) : 100) : 0; e = getenv("DM_MEM_PRETRIM_AVAIL_MB"); pre_avail = on ? (e ? atol(e) : 900) : 0;
        LOG("vk: memory watch: trim below %ld MB available or past %ld MB; early trim %s (game heap +%ld MB, under %ld MB available)", low, budget, on ? "on" : "off", pre_mb, pre_avail);
    }
    uint64_t now = now_ns(); int64_t hu = dm_ue_heap_used();
    io_pressure_sample(now);
    int jump = pre_mb > 0 && pbase >= 0 && hu - pbase >= (int64_t)pre_mb << 20 && now - last_trim >= 2000000000ull;
    if (!jump && now - last < 1000000000ull) return;
    int tick = now - last >= 1000000000ull; if (tick) last = now;
    long avail = 0, rss = 0; char ln[128]; FILE *f = fopen("/proc/meminfo", "re"); if (f) { while (fgets(ln, sizeof ln, f)) if (sscanf(ln, "MemAvailable: %ld", &avail) == 1) break; fclose(f); }
    long ranon = 0, rfile = 0, rshm = 0, vswap = 0;
    if ((f = fopen("/proc/self/status", "re"))) { while (fgets(ln, sizeof ln, f)) { sscanf(ln, "VmRSS: %ld", &rss); sscanf(ln, "RssAnon: %ld", &ranon); sscanf(ln, "RssFile: %ld", &rfile); sscanf(ln, "RssShmem: %ld", &rshm); sscanf(ln, "VmSwap: %ld", &vswap); } fclose(f); }
    uint64_t gpu = 0; lk_(&g_blkmu, LW_BLKMU); for (int i = 0; i < g_nblk; i++) if (g_blk[i].m) gpu += g_blk[i].size; pthread_mutex_unlock(&g_blkmu);
    gpu += atomic_load(&g_ring0_bytes) + atomic_load(&g_extra_ring_bytes);
    long ours = rss / 1024 + (long)(gpu >> 20);   // the process's resident memory and the backend's device memory, in MB
    int fall = 0;
    if (tick) {
        static int64_t pdrv = -1; static uint64_t last_gl; uint64_t bud = 0; int64_t drv = (int64_t)gpu_driver_usage(&bud);   // the driver's count (VK_EXT_memory_budget), 0 without it
        int64_t dh = hbase >= 0 ? hu - hbase : 0, db = dm_ue_heap_big() - bbase, ddrv = pdrv >= 0 ? drv - pdrv : 0; long drss = (rss - prss) / 1024, dav = (avail - pavail) / 1024;
        if (hbase >= 0) {
            fall = pre_mb > 0 && ((pavail > 0 && avail > 0 && dav <= -2 * pre_mb) || ddrv >= (int64_t)pre_mb << 20 || drss >= 2 * pre_mb);   // MemAvailable fell fast, the driver's GPU memory or the resident memory grew fast
            if ((dh >= 32 << 20 || dh <= -(64 << 20) || drss >= 64 || dav <= -128 || ddrv >= 64 << 20 || ddrv <= -(64 << 20)) && nline++ < 300)
                LOG("[mem] last second: game heap %+.0f MB (%.0f MB in use; big blocks %+.0f MB), resident %+ld MB (%ld MB: anonymous %ld, file %ld, shared %ld; swapped %ld), available %+ld MB (%ld MB), GPU driver %+.0f MB (%.0f MB), device %llu MB",
                    dh / 1048576.0, hu / 1048576.0, db / 1048576.0, drss, rss / 1024, ranon / 1024, rfile / 1024, rshm / 1024, vswap / 1024, dav, avail / 1024, ddrv / 1048576.0, drv / 1048576.0, (unsigned long long)(gpu >> 20));
        }
        if (now - last_gl >= 30000000000ull) {   // every 30 s: the driver's count against the backend's own allocations
            last_gl = now; int64_t all = atomic_load(&g_dm_live), blk = 0; lk_(&g_blkmu, LW_BLKMU); for (int i = 0; i < g_nblk; i++) if (g_blk[i].m) blk += g_blk[i].size; pthread_mutex_unlock(&g_blkmu);
            int64_t rg = atomic_load(&g_ring0_bytes), xr = atomic_load(&g_extra_ring_bytes), bi = atomic_load(&g_big_bytes), fb = atomic_load(&g_fbs_bytes), kept = (int64_t)g_xc_bytes;
            LOG("[mem] GPU memory: the driver counts %.0f MB (budget %.0f MB%s); the backend's allocations %.0f MB in %lld: image blocks %.0f, rings %.0f, extra rings %.0f (%.0f kept), big images %.0f, feedback snapshots %.0f, the rest %.0f",
                drv / 1048576.0, bud / 1048576.0, g_has_budget ? "" : ", no VK_EXT_memory_budget", all / 1048576.0, (long long)atomic_load(&g_dm_n), blk / 1048576.0, rg / 1048576.0, xr / 1048576.0, kept / 1048576.0,
                bi / 1048576.0, fb / 1048576.0, (all - blk - rg - xr - kept - bi - fb) / 1048576.0);
        }
        { static uint64_t pup, ptm, pring, pheld, pskip, pxn; uint64_t up = atomic_load(&g_up_cum), tm = atomic_load(&g_tex_made), rg = atomic_load(&g_up_ring_cum), xn = atomic_load(&g_xring_n);
          uint64_t pk = atomic_exchange(&g_xr_peak, atomic_load(&g_extra_ring_bytes)), held = g_reup_held, skip = g_reup_lv_skipped;
          if (hbase >= 0 && (up - pup >= (64ull << 20) || rg - pring >= (64ull << 20) || drss >= 64) && nline++ < 300)
              LOG("[mem] last second: texture uploads %.0f MB of storage, %.0f MB through the rings, %llu textures made; re-uploads held %llu, levels left out %llu; extra rings made %llu, at most %.0f MB at once",
                  (up - pup) / 1048576.0, (rg - pring) / 1048576.0, (unsigned long long)(tm - ptm), (unsigned long long)(held - pheld), (unsigned long long)(skip - pskip), (unsigned long long)(xn - pxn), pk / 1048576.0);
          pup = up; ptm = tm; pring = rg; pheld = held; pskip = skip; pxn = xn; }
        hbase = hu; bbase = dm_ue_heap_big(); prss = rss; pavail = avail; pdrv = drv;
    }
    int is_low = low > 0 && avail > 0 && avail / 1024 < low, over = budget > 0 && ours > budget;
    int near = pre_avail > 0 && avail > 0 && avail / 1024 < pre_avail;
    int early = (jump || fall) && now - last_trim >= 2000000000ull;
    if (early) pre_until = now + 10000000000ull;
    if (tick || early) pbase = hu;
    g_mem_low = is_low || over || near || now < pre_until;
    g_mem_tight = early || now < pre_until || (low > 0 && avail > 0 && avail / 1024 < low / 2);   // real pressure: a burst, or under half the low mark
    g_mem_real_low = is_low || over;   // DM_VK_RING_REUSE: kept rings go only when memory is really low (the bursts came from freeing and making the rings themselves)
    // A steady low (vk19.38): the dungeon sat at ~1 GB available, under the low mark and the budget, and the backend trimmed every 5 s, the most it may: every ring back to
    // its minimum (fresh mappings again, each page faulted on its first write: ~4,000 faults a frame on the render thread), textures re-made, malloc purged. Now a low that
    // holds without falling trims again only after DM_VK_TRIM_STEADY_S (default 60) seconds, or at once when MemAvailable fell 100 MB since the last trim; rings go back to their
    // minimum and malloc is purged only under real pressure (g_mem_tight). DM_VK_TRIM_STEADY_S=5: as before
    static long steady_s = -1; static long avail_at_trim; if (steady_s < 0) { const char *e = getenv("DM_VK_TRIM_STEADY_S"); steady_s = e && atol(e) > 0 ? atol(e) : 60; }
    uint64_t gap = early ? 2000000000ull : (avail / 1024 + 100 < avail_at_trim || g_mem_tight) ? 5000000000ull : (uint64_t)steady_s * 1000000000ull;
    if (!early && (!(is_low || over || near) || now - last_trim < gap)) {
        if (dm_ueheap_release && now - last_rel >= 60000000000ull) { last_rel = now; heap_release_kick(); }   // on the heap thread, not this one (it walks the heap under its locks)
        return;
    }
    static int fast_age = -1; if (fast_age < 0) { const char *e = getenv("DM_VK_EVICT_FAST"); fast_age = e && atoi(e) > 0 ? atoi(e) : 30; }
    atomic_store(&g_trim_age, early ? fast_age : 60);
    last_trim = now; avail_at_trim = avail / 1024; atomic_fetch_add(&g_trim_gen, 1); if (g_mem_tight && (!ring_reuse_on() || g_mem_real_low)) atomic_fetch_add(&g_ring_trim_gen, 1);   // DM_VK_RING_REUSE: rings stay unless memory is really low
    if (dm_ueheap_release) { last_rel = now; heap_release_kick(); }
    pthread_mutex_lock(&g_snapmu); for (int k = 0; k < SNAP_NC; k++) { while (g_snap_n[k]) free(g_snap_free[k][--g_snap_n[k]]); } g_snap_kept = 0; pthread_mutex_unlock(&g_snapmu);
    pthread_mutex_lock(&g_xcmu); while (g_nxc && (!ring_reuse_on() || g_mem_real_low)) { g_nxc--; vkDestroyBuffer(g_dev, g_xc[g_nxc].b, NULL); vkFreeMemory(g_dev, g_xc[g_nxc].m, NULL); g_xc_bytes -= g_xc[g_nxc].size;
        for (int k = 0; k < g_nxmap; k++) if (g_xmap[k].m == g_xc[g_nxc].m) { g_xmap[k] = g_xmap[--g_nxmap]; break; } } if (!g_nxc) g_xc_bytes = 0; pthread_mutex_unlock(&g_xcmu);   // only the rings let go (DM_VK_RING_REUSE keeps the others, and their bytes)
    if (g_mem_tight) mallopt(M_PURGE, 0);
    static int nl; if (nl++ < 60) LOG("vk: memory %s (%ld MB available, the process %ld MB resident + %llu MB device, the game heap %.0f MB): idle textures evicted, reuse pools emptied%s",
                                      jump ? "early trim: the game heap grew fast" : fall ? "early trim: MemAvailable fell, or GPU or resident memory grew, fast" : is_low ? "low" : over ? "over budget" : "near low (early trim)",
                                      avail / 1024, rss / 1024, (unsigned long long)(gpu >> 20), hu / 1048576.0, g_mem_tight ? ", rings back to their minimum, malloc purged" : "");
}
// mem_watch runs on a thread of its own (vk19.32), every 100 ms: run from the submissions it stopped with them, and FINAL FANTASY RESONANCE's killing loads stall the rendering
// (a frame of 54 NVN functions, then nothing): no line and no trim in the 5 s between the 115.7 MB read and the kill. The textures are still evicted at the next submission (tex_evict)
static void topo_tick(void);   // DM_TOPO_AFFINITY (below), once a second
static void *mem_watch_thread(void *a) { pthread_setname_np("dm-memwatch"); (void)a; for (int i = 0;; i++) { mem_watch(); if (i % 10 == 0) topo_tick(); usleep(100000); } return NULL; }
static void mem_watch_start(void) { static int started; if (started) return; started = 1; pthread_t t; if (pthread_create(&t, NULL, mem_watch_thread, NULL)) { started = 0; return; } pthread_detach(t); }
void tex_evict_hook(void) { mem_watch_start(); tex_evict(atomic_load(&g_trim_gen)); }   // the front end's low-memory signal (app/front_android.c): every ring back to RING_MIN at its next use
// DM_VK_RING_KEEP=1: a ring is not halved after a quiet stretch unless memory is low (g_mem_low). Halving and growing again later made a fresh mapping each time, every page of it
// faulted in on first write (bursts of thousands of minor faults while a stage streamed in). Default 0: as before
static int ring_keep(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_RING_KEEP"); v = e && atoi(e) != 0; if (v) LOG("DM_VK_RING_KEEP=1: rings shrink only while memory is low"); } return v; }
static void ring_fit(ctx_t *c) {
    c->last_total = c->total; if (c->total > c->win_max) c->win_max = c->total;
    VkDeviceSize want = c->ring0_size; uint64_t tg = atomic_load(&g_ring_trim_gen);
    if (c->trim_seen != tg) { c->trim_seen = tg; want = RING_MIN; c->win_max = 0; c->nwin = 0; }
    else if (c->last_total > c->ring0_size && c->ring0_size < RING_MAX) { while (want < c->last_total && want < RING_MAX) want *= 2;
                                                                         if (!ring_reuse_on()) { if (atomic_load(&g_ring0_bytes) + (want - c->ring0_size) > RING_CAP) want = c->ring0_size; }
                                                                         else while (want > c->ring0_size && atomic_load(&g_ring0_bytes) + (want - c->ring0_size) > RING_CAP + (uint64_t)NCTX * RING_MIN) want /= 2; }   // DM_VK_RING_REUSE: the largest that fits, above each context's minimum
    else if (++c->nwin >= 120) { if (c->win_max * 4 < c->ring0_size && c->ring0_size > RING_MIN && (!ring_keep() || g_mem_low)) want = c->ring0_size / 2; c->win_max = 0; c->nwin = 0; }   // DM_VK_RING_KEEP=1: halved only while memory is low
    if (want == c->ring0_size) return;
    vkDestroyBuffer(g_dev, c->ring0, NULL); vkFreeMemory(g_dev, c->ringm0, NULL); atomic_fetch_sub(&g_ring0_bytes, c->ring0_size);
    c->ring0 = buf_new(want, ALL_BUF_USAGE, &c->ringm0, (void **)&c->ringp0); c->ring0_size = want; atomic_fetch_add(&g_ring0_bytes, want); g_ring_new++; g_ring_new_bytes += want;
}
// DM_VK_CTX_PICK (vk19.57, Path A S0; the default from vk19.58), the Metal host's choice (nvn_mtl.m mtl_sub_begin: any finished allocator, a new one if none, a wait only at the cap). The
// contexts were taken round robin, and the one whose turn it was could be held by a later submission still reading its ring (GPU-written data used in place, gw_find): the
// recorder waited 8.4-8.6 ms a frame for that reader, usually the newest submission, so CPU and GPU took turns. Now the least recently used context whose submission is done and
// that no later one reads is taken; with none, a new one is made, up to DM_VK_CTX_MAX (default 32); only at the cap the recorder waits, for the context that frees first. Ring and
// hold rules are unchanged. A context taken again ends the GPU-written copies forwarded into its ring (DM_VK_GW_FORWARD counted on round robin). Per 600 frames: picks free at
// once, contexts made, waits by reason (its own submission, a later reader, at the cap) and the high-water count
// On by default from vk19.58 (Run X, resolution pinned at 60%, ABAB on one install: 23.6 / 23.9 -> 48.5 / 50.0 fps, the context wait 8 -> 0 ms a frame, GPU 25.7 -> 14.0 ms);
// DM_VK_CTX_PICK=0 takes the contexts round robin again
static int ctx_pick_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_CTX_PICK"); v = !(e && atoi(e) == 0);
    if (v) LOG("vk: DM_VK_CTX_PICK on (the default): a free submission context is taken, more are made up to DM_VK_CTX_MAX"); else LOG("vk: DM_VK_CTX_PICK=0: submission contexts taken round robin"); } return v; }
static uint64_t g_pick_free, g_pick_grow, g_pick_wait, g_pick_wait_fence, g_pick_wait_hold; static int g_pick_hw;
static void ctx_pick(void) {
    static int cap; if (!cap) { const char *e = getenv("DM_VK_CTX_MAX"); cap = e && atoi(e) > 0 ? atoi(e) : NCTX_MAX; if (cap > NCTX_MAX) cap = NCTX_MAX; if (cap < NCTX) cap = NCTX; }
    lk_(&g_mu, LW_MU);
    int best = -1, soon = -1; uint64_t soon_at = UINT64_MAX;
    for (int k = 0; k < NCTX; k++) { ctx_t *c = &g_ctx[k]; if (c == C) continue;   // the one just committed: its fence is not done yet anyway
        uint64_t f = c->fence_val > c->hold ? c->fence_val : c->hold;
        if (f <= g_done && !c->resetting) { if (best < 0) best = k;
                           else if (!ring_reuse_on()) { if (c->fence_val < g_ctx[best].fence_val) best = k; }
                           else if (c->reset_done > g_ctx[best].reset_done || (c->reset_done == g_ctx[best].reset_done && c->fence_val > g_ctx[best].fence_val)) best = k; }   // DM_VK_RING_REUSE: the most recent
        else if (f < soon_at) { soon_at = f; soon = k; } }
    int grow = best < 0 && NCTX < cap;
    pthread_mutex_unlock(&g_mu);
    // One choice: a free context, else a new one, else a wait at the cap. The ring-reuse upkeep below used to sit between the first and the other two (H0.5), which broke the
    // chain: with RING_REUSE off every free pick was then overridden by the wait branch (the busy context that frees first: the previous submission, CPU and GPU in turns;
    // FINAL FANTASY XII on main waited 22 ms a frame), with it on no context was ever made or waited for
    if (best >= 0) { g_ci = best; g_pick_free++; }
    else if (grow) { ctx_init(NCTX); lk_(&g_mu, LW_MU); g_ci = NCTX; NCTX++; pthread_mutex_unlock(&g_mu); g_pick_grow++; if (NCTX > g_pick_hw) g_pick_hw = NCTX;
                     static int nl; if (nl++ < 12) LOG("vk: DM_VK_CTX_PICK: no free context: context %d made", g_ci); }
    else { g_ci = soon >= 0 ? soon : (g_ci + 1) % NCTX; g_pick_wait++;   // at the cap: the one that frees first (the wait itself is counted below, as before)
           if (g_ctx[g_ci].hold > g_ctx[g_ci].fence_val) g_pick_wait_hold++; else g_pick_wait_fence++; }
    if (ring_reuse_on()) {   // DM_VK_RING_REUSE: contexts unused for 300 submissions give their grown ring back (checked every 64 submissions)
        static uint64_t n; if (!(++n & 63)) for (int k = 0; k < NCTX; k++) { ctx_t *c = &g_ctx[k];
            if (k == g_ci || c == C || c->ring0_size <= RING_MIN || c->h05_last + 300 > g_fence) continue;
            lk_(&g_mu, LW_MU); int idle = (c->fence_val > c->hold ? c->fence_val : c->hold) <= g_done && c->reset_done; pthread_mutex_unlock(&g_mu); if (!idle) continue;
            vkDestroyBuffer(g_dev, c->ring0, NULL); vkFreeMemory(g_dev, c->ringm0, NULL); atomic_fetch_sub(&g_ring0_bytes, c->ring0_size);
            c->ring0 = buf_new(RING_MIN, ALL_BUF_USAGE, &c->ringm0, (void **)&c->ringp0); c->ring0_size = RING_MIN; atomic_fetch_add(&g_ring0_bytes, RING_MIN); c->win_max = 0; c->nwin = 0; g_h05_shrunk++; }
    }
    if (g_pick_hw < NCTX) g_pick_hw = NCTX;
    lk_(&g_gwmu, LW_GWMU);   // copies forwarded into this context's ring are its old submission's: forgotten
    for (int i = 0; i < g_ngw; i++) if (g_gw[i].ffence && g_gw[i].fci == g_ci) g_gw[i].ffence = 0;
    pthread_mutex_unlock(&g_gwmu);
}
void mtl_sub_begin(void) {
    vk_init(); g_sub_up_bytes = 0;
    { void topo_self_tick(void); topo_self_tick(); }   // DM_TOPO_SELF: the recording thread places itself
    { extern _Atomic int g_rec_tid; static int cand, streak; int t = (int)gettid();   // the recording thread (DM_TOPO_AFFINITY): the one that made the last 30 submissions in a row
      if (t != cand) { cand = t; streak = 0; } if (++streak >= 30 && atomic_load_explicit(&g_rec_tid, memory_order_relaxed) != t) atomic_store(&g_rec_tid, t); }
    if (zc_mode()) zc_reap(g_done);
    ds_last_newcb();   // a new command buffer: no state recorded in it yet
    tog_latch();   // the run-time switches, for this whole submission
    ts_tick();   // the range store's upkeep (vk19.74c)
    { int cpu = sched_getcpu(); if (cpu >= 0 && cpu < 8) g_rcore[cpu]++; }
    if (C && ring_reuse_on()) { VkDeviceSize ov = C->total > C->ring0_size ? C->total - C->ring0_size : 0; g_h05_over -= g_h05_over / 8; if (ov > g_h05_over) g_h05_over = ov; }   // DM_VK_RING_REUSE: the last submission's overflow
    uint64_t w0 = now_ns();
    if (ctx_pick_on()) ctx_pick();
    else g_ci = (g_ci + 1) % NCTX;
    ctx_t *c = &g_ctx[g_ci];
    lk_(&g_mu, LW_MU);
    { uint64_t t1 = now_ns(); while (c->fence_val > g_done) pthread_cond_wait(&g_cv, &g_mu); uint64_t t2 = now_ns(); while (c->hold > g_done) pthread_cond_wait(&g_cv, &g_mu);
      atomic_fetch_add(&g_wait_fence_ns, t2 - t1); atomic_fetch_add(&g_wait_hold_ns, now_ns() - t2); }   // its own submission still running / a later one still reading its ring
    while (c->resetting) pthread_cond_wait(&g_cv, &g_mu);   // its pools still being reset (DM_VK_EARLY_DONE): ctx_pick avoids it, a wait at the cap may not
    pthread_mutex_unlock(&g_mu);
    lk_(&g_gwmu, LW_GWMU); c->hold = 0; pthread_mutex_unlock(&g_gwmu);
    atomic_fetch_add(&g_wait_ns, now_ns() - w0); atomic_fetch_add(&g_wait_cum, now_ns() - w0);
    ring_fit(c); c->h05_last = g_fence;
    C = c;
    { void tex_evict_hook(void); tex_evict_hook(); } C->used = 0; C->up_used = 0; g_ndc = 0; g_wgen++; g_nwlog = 0; g_h0_gen[0]++; g_h0_sub_gen0 = g_wgen; g_h0_ctx_used |= 1u << (g_ci & 31); C->nh0ip = 0;
    if (h0_on()) { uint64_t gb = 0; int gn; lk_(&g_gwmu, LW_GWMU); gn = g_ngw; for (int i = 0; i < g_ngw; i++) gb += g_gw[i].size; pthread_mutex_unlock(&g_gwmu);   // H0: GPU-written entries live now
      g_h0_gw_n += (uint64_t)gn; g_h0_gw_b += gb; if ((uint64_t)gn > g_h0_gw_nmax) g_h0_gw_nmax = (uint64_t)gn; if (gb > g_h0_gw_bmax) g_h0_gw_bmax = gb; } C->ring = C->ring0; C->ringp = C->ringp0; C->ring_size = C->ring0_size; C->total = 0;
    g_bvgen++;   // texel buffer views are this submission's (bview)
    C->ntr = C->ndraw = C->ndisp = C->max_groups = 0; memset(&C->heavy, 0, sizeof C->heavy); C->plog = 0;
    if (!C->reset_done) for (int i = 0; i < C->ndp; i++) VKCHECK(vkResetDescriptorPool(g_dev, C->dps[i], 0));   // normally done by the completion thread (ctx_complete)
    C->idp = 0;
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL};
    if (!C->reset_done) { VKCHECK(vkResetCommandPool(g_dev, C->pool, 0)); atomic_fetch_add(&g_cpr_rec, 1); }   // both command buffers at once, their memory kept; normally done already by the completion thread (counted: work on the recording thread)
    C->reset_done = 0;
    VKCHECK(vkBeginCommandBuffer(C->cb, &bi)); VKCHECK(vkBeginCommandBuffer(C->up, &bi)); C->nsec = 0; C->prim = VK_NULL_HANDLE;   // the pool's reset reset its secondaries too
    barrier_all(C->up); g_vbar_epoch++;   // after everything submitted before (see mtl_sub_commit: no semaphore wait between the game's submissions)
    vkCmdResetQueryPool(C->up, C->qp, 0, 3); vkCmdWriteTimestamp(C->up, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, C->qp, 0);
    if (!C->oq_init) { vkCmdResetQueryPool(C->up, C->oq, 0, NOQ); C->oq_init = 1; } else if (C->noq) vkCmdResetQueryPool(C->up, C->oq, 0, C->noq);   // only those used last time
    C->noq = 0; g_occ_over = 0;
    if (!C->tq_init) { vkCmdResetQueryPool(C->up, C->tq, 0, NTQ); C->tq_init = 1; } else if (C->ntq) vkCmdResetQueryPool(C->up, C->tq, 0, C->ntq);
    C->ntq = 0;
}
void mtl_sub_index(const uint64_t *h, int n) { (void)h; (void)n; }
void mtl_exec_rest(const uint64_t *handles, int n) { g_rest = handles; g_nrest = n; }   // the handles after the one replayed next (rt_hold_scan)
// A buffer copy is made on the CPU as the command list is walked (the in-order writes, wlog). Where the source holds bytes the GPU wrote and has not put back in guest memory
// yet (gw_find), the copy is made on the GPU from them into the ring instead, and the destination becomes a GPU-written range like the source (gw_add, back at completion)
// DM_VK_GW_PAD (Harvest Moon heavy scene; off by default): a buffer copy of GPU-written bytes (mtl_sub_copy) of at most 256 bytes registers its destination padded to the
// 256-byte-aligned window 64 bytes around it (within its pool), the padding taken from the guest's bytes, when nothing else GPU-written lies in that window. hm9: 83 composes
// a frame came from 20-byte indirect-argument reads around a 4-byte GPU-written instance count (an exact copy's entry), each ending the render pass. With the pad the read fits
// the entry and is bound in place. The entry has a snapshot (the guest's bytes), so it goes back by changed words (post kind 5) and the CPU's later rewrites of the padding are
// patched into later reads (gw_cpu_dirty), as for any dispatch range. Counted in the per-600 line 'vk: DM_VK_GW_PAD'
static int gw_pad_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_GW_PAD"); v = e && atoi(e) != 0; if (v) LOG("vk: DM_VK_GW_PAD=1: small copies of GPU-written bytes padded with the guest's bytes around them"); } return v; }
static uint64_t g_pad_n, g_pad_into, g_pad_over, g_pad_why[4];   // windows made, copies into a window, earlier submissions' parts put into new windows; not padded: 1 no pool, 2 this submission's other GPU writes, 3 no room
static uint64_t pool_range(uint64_t addr, uint64_t *size);
static int gw_any(uint64_t a, uint64_t n) { int r = 0; if (!gw_pg_any(a, n)) return 0; lk_(&g_gwmu, LW_GWMU); for (int i = 0; i < g_ngw && !r; i++) r = !GW_MISS(i, a, n); pthread_mutex_unlock(&g_gwmu); return r; }
void mtl_sub_copy(uint64_t src, uint64_t dst, uint64_t n) {
    memmove((void *)dst, (const void *)src, n); wlog_add(dst, n);
    if (!n || !g_ngw || !gw_any(src, n)) return;
    if (chain_snap_mode()) { lk_(&g_gwmu, LW_GWMU); gw_snap_follow((const uint8_t *)(uintptr_t)dst, n, (const uint8_t *)(uintptr_t)dst, NULL, 0); pthread_mutex_unlock(&g_gwmu); g_cs_b2b++; }   // the CPU-side copy of GPU-written bytes above is stale, not a CPU rewrite: the live snapshots over dst take it too (gw_cpu_dirty would patch it over the GPU's bytes)
    pass_end();
    g_h0_caller = H0C_COPY; VkBuffer sb; VkDeviceSize so; if (!gw_find(src, n, 16, &sb, &so, NULL)) { memmove((void *)dst, (const void *)src, n); return; }   // complete now (gw_find waited for it): in guest memory
    if (gw_pad_on() && n <= 256) {   // DM_VK_GW_PAD: small copies into a padded 4 KB window of the guest's bytes, so reads of the records around them fit (no compose, no pass break)
        uint64_t psz = 0, base = pool_range(dst, &psz), lo = dst & ~(uint64_t)4095, hi = lo + 4096;
        if (base && lo < base) lo = base; if (base && hi > base + psz) hi = base + psz;
        int why = !base || dst < lo || dst + n > hi ? 1 : 0;
        if (!why) {
            lk_(&g_gwmu, LW_GWMU);
            for (int i = g_ngw - 1; i >= 0; i--) { gw_t *e = &g_gw[i];   // this submission's window already holding dst: the copy goes into it
                if (e->pad && e->fence == g_fence + 1 && dst >= e->addr && dst + n <= e->addr + e->size) {
                    VkBuffer eb = e->buf; VkDeviceSize eo = e->off + (dst - e->addr);
                    if (e->snap) memcpy((uint8_t *)(uintptr_t)e->snap + (dst - e->addr), (const void *)(uintptr_t)dst, n);   // the CPU-side copy above is not a CPU rewrite
                    pthread_mutex_unlock(&g_gwmu);
                    VkBufferCopy bc = {so, eo, n}; vkCmdCopyBuffer(C->cb, sb, eb, 1, &bc); barrier_xfer(C->cb);
                    atomic_fetch_add(&g_gw_b2b, 1); g_pad_into++; return;
                } }
            for (int i = 0; i < g_ngw && !why; i++) if (!GW_MISS(i, lo, hi - lo) && g_gw[i].fence == g_fence + 1) why = 2;   // this submission's other GPU writes there: not padded
            pthread_mutex_unlock(&g_gwmu);
        }
        VkDeviceSize ro = !why ? ring_alloc(hi - lo, 256) : VK_WHOLE_SIZE; uint8_t *snap = ro != VK_WHOLE_SIZE ? snap_alloc(hi - lo) : NULL;
        if (!why && !snap) why = 3;
        if (!why) {
            VkBuffer rb = C->ring; memcpy(C->ringp + ro, (const void *)(uintptr_t)lo, hi - lo); memcpy(snap, (const void *)(uintptr_t)lo, hi - lo);
            lk_(&g_gwmu, LW_GWMU);   // an earlier submission's GPU writes in the window (still running): their bytes over the guest's, at this submission's start, as gw_compose does
            for (int i = 0; i < g_ngw; i++) { if (GW_MISS(i, lo, hi - lo)) continue; gw_t *e = &g_gw[i]; if (e->fence == g_fence + 1) continue;
                uint64_t a0 = e->addr > lo ? e->addr : lo, a1 = e->addr + e->size < hi ? e->addr + e->size : hi;
                VkBufferCopy bc = {e->off + (a0 - e->addr), ro + (a0 - lo), a1 - a0}; vkCmdCopyBuffer(C->up, e->buf, rb, 1, &bc); C->up_used = 1;
                if (gw_cpu_dirty(e, a0, a1)) gw_patch(e, a0, a1, rb, ro + (a0 - lo));
                ctx_t *o = &g_ctx[e->ci]; if (o->hold < g_fence + 1) o->hold = g_fence + 1; xr_raise(e->buf, g_fence + 1); g_pad_over++; }
            pthread_mutex_unlock(&g_gwmu);
            VkBufferCopy bc = {so, ro + (dst - lo), n}; vkCmdCopyBuffer(C->cb, sb, rb, 1, &bc); barrier_xfer(C->cb);
            g_gw_add_pad = 1; gw_add(lo, hi - lo, rb, ro, C->ringp + ro, g_fence + 1, g_ci, snap, 1); g_gw_add_pad = 0;
            post_add((post_t){5, (void *)(uintptr_t)lo, 0, 0, (uint64_t)(uintptr_t)snap, 0, C->ringp + ro, hi - lo});   // back by changed words: the CPU's writes to the window stay
            atomic_fetch_add(&g_gw_b2b, 1); g_pad_n++; return;
        }
        if (snap) snap_free(snap, hi - lo);
        g_pad_why[why]++;
    }
    VkDeviceSize ro = ring_alloc(n, 16); if (ro == VK_WHOLE_SIZE) { RED_ONCE(93, "a buffer copy of GPU-written bytes found no room in the ring: copied from guest memory"); return; }
    VkBufferCopy bc = {so, ro, n}; vkCmdCopyBuffer(C->cb, sb, C->ring, 1, &bc); barrier_xfer(C->cb);
    gw_add(dst, n, C->ring, ro, C->ringp + ro, g_fence + 1, g_ci, NULL, 1); post_add((post_t){3, (void *)(uintptr_t)dst, 0, 0, 0, 0, C->ringp + ro, n});
    atomic_fetch_add(&g_gw_b2b, 1);
}
void mtl_sub_event(volatile uint32_t *val, uint32_t mode, uint32_t v) { post_add((post_t){1, (void *)val, mode, v}); }
void mtl_sub_counter(uint64_t addr, uint32_t type) { post_add((post_t){2, NULL, 0, 0, addr, type}); }
static void mtl_exec_(const uint8_t *p);
void mtl_exec(const uint8_t *p) { uint64_t t0 = now_ns(); mtl_exec_(p); atomic_fetch_add(&g_rec_ns, now_ns() - t0); }
// NVN's initial command buffer state, as the Metal backend's state_reset: games set some of it only by state objects or never (Metroid Dread never sets the stencil value mask,
// which starts at 0xff; left at 0 its G-buffer draws, stencil test "not equal 2", failed everywhere and the level's opaque geometry stayed black)
static void state_defaults(void) {
    for (int i = 0; i < 6; i++) g_bgen[i] = g_bgen_next++;   // the bindings may change wholesale
    for (int i = 0; i < 8; i++) S.blend[i] = (blend_st){i, NVN_BLEND_FUNC_ONE, NVN_BLEND_FUNC_ZERO, NVN_BLEND_FUNC_ONE, NVN_BLEND_FUNC_ZERO, NVN_BLEND_EQUATION_ADD, NVN_BLEND_EQUATION_ADD};
    S.color = (color_st){0, NVN_LOGIC_OP_COPY, NVN_ALPHA_FUNC_ALWAYS}; S.chmask.mask = 0xffffffffu;
    S.poly = (poly_st){NVN_FACE_NONE, NVN_FRONT_FACE_CCW, NVN_POLYGON_MODE_FILL, 0};
    S.ds = (ds_st){0, 1, NVN_DEPTH_FUNC_LESS, 0, NVN_STENCIL_FUNC_ALWAYS, NVN_STENCIL_FUNC_ALWAYS, 1, 1, 1, 1, 1, 1};
    S.st_vmask[0] = S.st_vmask[1] = S.st_wmask[0] = S.st_wmask[1] = 0xff; S.dr[1] = 1; S.line_w = 1;
}
static void mtl_exec_(const uint8_t *p) {
    static int inited; if (!inited) { inited = 1; state_defaults(); }
    for (;;) {
        op_hdr h; memcpy(&h, p, sizeof h);
        const uint8_t *a = p + 8;
        switch (h.op) {
            case OP_END: return;
            case OP_JUMP: { const uint8_t *t; memcpy(&t, a, 8); p = t; continue; }
            case OP_CALL: { const uint8_t *t; memcpy(&t, a, 8); if (g_nrs < 32) g_rs[g_nrs] = p + h.size; g_nrs++; mtl_exec_(t); g_nrs--; break; }   // g_rs: where the replay returns (rt_hold_scan)
            case OP_SIGNAL_EVENT: { volatile uint32_t *val; uint32_t m, v; memcpy(&val, a, 8); memcpy(&m, a + 8, 4); memcpy(&v, a + 12, 4); mtl_sub_event(val, m, v); break; }
            case OP_REPORT_COUNTER: { uint64_t t, ad; memcpy(&t, a, 8); memcpy(&ad, a + 8, 8); if (t == NVN_COUNTER_TYPE_SAMPLES_PASSED) occ_report(ad); else mtl_sub_counter(ad, (uint32_t)t); break; }
            case OP_RESET_COUNTER: { uint64_t t; memcpy(&t, a, 8); if (t == NVN_COUNTER_TYPE_SAMPLES_PASSED) occ_reset(); break; }
            case OP_FENCE_SYNC: { sync_t *s; memcpy(&s, a, 8); s->fence = g_fence + 1; s->fenced = 1; break; }
            case OP_CLEAR_BUF: { const p_clear_buf *cb = (const p_clear_buf *)a; for (uint64_t o = 0; o + 4 <= cb->size; o += 4) memcpy((uint8_t *)cb->addr + o, &cb->value, 4); wlog_add(cb->addr, cb->size); break; }
            case OP_COPY_B2B: { uint64_t s, d, n; memcpy(&s, a, 8); memcpy(&d, a + 8, 8); memcpy(&n, a + 16, 8); mtl_sub_copy(s, d, n); break; }
            case OP_UPDATE_UB: { uint64_t d, n; memcpy(&d, a, 8); memcpy(&n, a + 8, 8); memcpy((void *)d, a + 16, n); wlog_add(d, n); break; }
            case OP_SET_RT: set_rt(p, (const p_set_rt *)a, h.size); break;
            case OP_CLEAR_COLOR: clear_color((const p_clear_color *)a); break;
            case OP_CLEAR_DS: clear_ds((const p_clear_ds *)a); break;
            case OP_DRAW: { uint64_t t0 = stat_ns(); p_draw t; draw(draw_rec(a, h.size, &t), NULL); g_t_draw += stat_ns() - t0; break; }
            case OP_DRAW_IDX: { uint64_t t0 = stat_ns(); p_draw_idx t; draw(NULL, draw_idx_rec(a, h.size, &t)); g_t_draw += stat_ns() - t0; break; }
            case OP_DISPATCH: { uint64_t t0 = stat_ns(); dispatch((const p_dispatch *)a); g_t_disp += stat_ns() - t0; break; }
            case OP_COPY_B2T: copy_b2t((const p_copy_b2t *)a); break;
            case OP_COPY_T2T: copy_t2t((const p_copy_t2t *)a); break;
            case OP_COPY_T2B: copy_t2b((const p_copy_t2t *)a); break;
            case OP_CLEAR_TEX: clear_tex((const p_clear_tex *)a); break;
            case OP_BARRIER: if (barrier_keeps_pass(p + h.size)) { P.merged = 1; g_nbar_kept++; } else pass_end(); break;   // see barrier_keeps_pass
            case OP_SCISSOR: S.sc = *(const p_scissor *)a; S.has_sc = 1; break;
            case OP_BIND_PROGRAM: { const p_bind_program *b = (const p_bind_program *)a; if (b->stages & 0x20) S.cprog = b->prog; for (int st = 0; st < 5; st++) if (b->stages >> st & 1) S.bst[st] = b->prog; break; }
            case OP_BIND_STATE: {
                const p_bind_state *b = (const p_bind_state *)a; blend_st bs;
                switch (b->kind) {
                    case 0: memcpy(&bs, b->bytes, sizeof bs); S.blend[bs.target] = bs; break;
                    case 1: memcpy(&S.color, b->bytes, sizeof S.color); break; case 2: memcpy(&S.chmask, b->bytes, sizeof S.chmask); break;
                    case 3: memcpy(&S.ms, b->bytes, sizeof S.ms); break; case 4: memcpy(&S.poly, b->bytes, sizeof S.poly); break; case 5: memcpy(&S.ds, b->bytes, sizeof S.ds); break;
                }
                break;
            }
            case OP_BIND_VATTR: { const p_bind_vattr *b = (const p_bind_vattr *)a; memcpy(S.attr, b->st, 4 * b->n); S.nattr = b->n; break; }
            case OP_BIND_VSTREAM: { const p_bind_vstream *b = (const p_bind_vstream *)a; memcpy(S.stream, b->st, 8 * b->n); S.nstream = b->n; break; }
            case OP_BIND_VBUF: case OP_BIND_VBUF1: { p_bind_vbuf t1; const p_bind_vbuf *b = vbuf_rec(h.op, a, &t1); if (b->first >= 0 && b->first + b->n <= 16) memcpy(&S.vbuf[b->first], b->r, sizeof *b->r * b->n); break; }
            case OP_BIND_UBO: case OP_BIND_UBO_SMALL: { p_bind_buf t1; const p_bind_buf *b = bind_buf_rec(h.op, a, &t1); if (b->stage < 6 && b->index < 16) S.ubo[b->stage][b->index] = *b; break; }
            case OP_BIND_SSBO: { const p_bind_buf *b = (const p_bind_buf *)a; if (b->stage < 6 && b->index < 16) { S.ssbo[b->stage][b->index] = *b; g_bgen[b->stage] = g_bgen_next++; } break; }
            case OP_BIND_TEX: case OP_BIND_TEX1: case OP_BIND_SAMP: case OP_BIND_IMG: {
                p_bind_handles t1; const p_bind_handles *b = bind_rec(h.op, a, &t1); uint64_t hh[32]; memcpy(hh, b->h, 8 * (b->n < 32 ? b->n : 32));
                if (b->defer && b->src) memcpy(hh, (const void *)(uintptr_t)b->src, 8 * (b->n < 32 ? b->n : 32));   // BindTexturesDeferred: the handles as they are now
                uint64_t *dst = h.op == OP_BIND_TEX || h.op == OP_BIND_TEX1 ? S.tex[b->stage] : h.op == OP_BIND_SAMP ? S.samp[b->stage] : S.img[b->stage];
                int lim = h.op == OP_BIND_TEX || h.op == OP_BIND_TEX1 ? NTEX : h.op == OP_BIND_SAMP ? 128 : 32;
                if (b->stage < 6 && b->first >= 0 && b->first + b->n <= lim) { memcpy(dst + b->first, hh, 8 * b->n); g_bgen[b->stage] = g_bgen_next++; }
                break;
            }
            case OP_STENCIL: { const p_stencil *b = (const p_stencil *)a; for (int f = 0; f < 2; f++) if (b->face & (1 << f)) { if (b->which == 0) S.st_vmask[f] = b->value_mask; else if (b->which == 1) S.st_wmask[f] = b->mask; else S.st_ref[f] = b->ref; } break; }
            case OP_POLY_OFFSET: S.po = *(const p_poly_offset *)a; break;
            case OP_VIEWPORTS: { const p_viewports *b = (const p_viewports *)a; for (int i = 0; i < b->n; i++) memcpy(S.vp[b->first + i], b->v[i], 16); S.has_vp = 1; break; }
            case OP_DEPTH_RANGE: { const p_depth_range *b = (const p_depth_range *)a; S.dr[0] = b->n; S.dr[1] = b->f; break; }
            case OP_BLEND_COLOR: memcpy(S.blend_color, ((const p_blend_color *)a)->rgba, 16); break;
            case OP_SET_TEX_POOL: case OP_SET_SAMP_POOL: { int nvn_dpool_base(const void *pool, int tex); int tex = h.op == OP_SET_TEX_POOL, bs = nvn_dpool_base(((const p_set_pool *)a)->pool, tex);
                if (tex) S.tex_base = bs; else S.samp_base = bs; break; }
            case OP_LINE_WIDTH: S.line_w = ((const p_line_width *)a)->w; break;
            case OP_PRIM_RESTART: S.restart = *(const p_prim_restart *)a; break;
            case OP_ALPHA_REF: break;   // the test itself is reported at the draw (not implemented)
            case OP_DRAW_TEX: RED_ONCE(49, "DrawTexture is not implemented: not drawn"); break;
            case OP_COPY_STRIDE: { const p_copy_stride *cs = (const p_copy_stride *)a; if (cs->image) S.copy_img_stride = cs->stride; else S.copy_stride = cs->stride; break; }
            default: break;
        }
        p += h.size;
    }
}
static void occ_flush(void);
uint64_t mtl_sub_commit(void) {
    if (g_ntz) texu_zero_commit();   // (vk-t10 diagnostics)
    pass_end(); occ_flush();
    vkCmdWriteTimestamp(C->cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, C->qp, 1);
    barrier_all(C->up);   // the uploads, clears and copies at the start are done and visible before the submission's own work (nothing orders two command buffers of one batch)
    vkCmdWriteTimestamp(C->up, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, C->qp, 2);   // the upload command buffer's own GPU time (g_up_gpu_ns)
    VKCHECK(vkEndCommandBuffer(C->cb)); VKCHECK(vkEndCommandBuffer(C->up));
    uint64_t fence = ++g_fence; C->fence_val = fence;
    VkCommandBuffer cbs[2] = {C->up, C->cb};
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 2, cbs, 0, NULL};
    VKCHECK(vkResetFences(g_dev, 1, &C->fence));
    VkTimelineSemaphoreSubmitInfo ts = {VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO}; uint64_t wv, sv; VkPipelineStageFlags wst = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    lk_(&g_qmu, LW_QMU);
    // The game's submissions are ordered among themselves by the barrier each begins with (one queue: a barrier waits for everything submitted before it). Each still signals the
    // timeline (the present waits for the frame's last one), but waits on it only after a present: the Mali-G57 did not keep a present's blit in order with what followed, and a
    // semaphore wait on every one of Unreal's ~9 submissions a frame left the Adreno 825 idle between them (GPU 10% busy at 19 fps)
    if (g_tl) { sv = ++g_tlv; ts.signalSemaphoreValueCount = 1; ts.pSignalSemaphoreValues = &sv; si.pNext = &ts; si.signalSemaphoreCount = 1; si.pSignalSemaphores = &g_tl;
                if (g_after_present) { wv = sv - 1; ts.waitSemaphoreValueCount = 1; ts.pWaitSemaphoreValues = &wv; si.waitSemaphoreCount = 1; si.pWaitSemaphores = &g_tl; si.pWaitDstStageMask = &wst; }
                g_after_present = 0; }
    uint64_t ts0 = now_ns(); VkResult r = vkQueueSubmit(g_q, 1, &si, C->fence); g_t_sub += now_ns() - ts0; pthread_mutex_unlock(&g_qmu);
    { static uint64_t nsub; nsub++; extern uint64_t g_nsub_stat; g_nsub_stat = nsub; }
    if (r != VK_SUCCESS) { trace_dump("submission failed"); die("Vulkan: submission failed (%d)", (int)r); }
    lk_(&g_mu, LW_MU); g_pend[g_npend++] = g_ci; pthread_cond_broadcast(&g_cv); pthread_mutex_unlock(&g_mu);
    return fence;
}
uint64_t mtl_last_fence(void) { return g_fence; }
// The record thread (hle_nvn.c, DM_VK_REC_THREAD): fence values handed out as submissions are queued (they commit in that order: ++g_fence in mtl_sub_commit), and the queue's
// drain, which whatever waits for all the GPU's work runs first (mtl_finish)
static _Atomic uint64_t g_fence_q; static void (*g_rec_drain)(void);
uint64_t mtl_fence_queued(void) { uint64_t q = atomic_load(&g_fence_q); return q > g_fence ? q : g_fence; }
void mtl_fence_queued_add(void) { uint64_t f = g_fence, q = atomic_load(&g_fence_q); if (q < f) atomic_compare_exchange_strong(&g_fence_q, &q, f); atomic_fetch_add(&g_fence_q, 1); }
void mtl_rec_drain_hook(void (*f)(void)) { g_rec_drain = f; }
uint64_t mtl_present_count(void) { return atomic_load(&g_presents); }
void mtl_wait_committed(uint64_t fence) { lk_(&g_mu, LW_MU); while (g_fence < fence) pthread_cond_wait(&g_cv, &g_mu); pthread_mutex_unlock(&g_mu); }   // until the record thread has submitted it (mtl_sub_commit broadcasts g_cv)
int mtl_wait(uint64_t fence, uint64_t timeout_ns) {
    uint64_t w0 = now_ns(); int waited = 0;
    lk_(&g_mu, LW_MU); waited = g_done < fence; if (waited) { atomic_fetch_add(&g_gw8_lag, g_fence - fence); atomic_fetch_add(&g_gw8_pend, fence - g_done); }
    if (timeout_ns == UINT64_MAX || timeout_ns > 60ull * 1000000000ull) { while (g_done < fence) pthread_cond_wait(&g_cv, &g_mu); }
    else {
        struct timespec t; clock_gettime(CLOCK_REALTIME, &t); uint64_t ns = (uint64_t)t.tv_nsec + timeout_ns; t.tv_sec += (time_t)(ns / 1000000000ull); t.tv_nsec = (long)(ns % 1000000000ull);
        while (g_done < fence) if (pthread_cond_timedwait(&g_cv, &g_mu, &t)) break;
    }
    int ok = g_done >= fence; pthread_mutex_unlock(&g_mu);
    if (waited) { atomic_fetch_add(&g_gw8_ns, now_ns() - w0); atomic_fetch_add(&g_gw8_n, 1); }
    return ok;
}
void mtl_finish(void) {
    if (g_rec_drain) g_rec_drain(); mtl_wait(g_fence, UINT64_MAX);
    lk_(&g_mu, LW_MU);   // and the pool resets that follow a fence (DM_VK_EARLY_DONE): a drain (hibernation's snapshot) leaves no context half reset
    for (int k = 0; k < NCTX; k++) while (g_ctx[k].resetting) pthread_cond_wait(&g_cv, &g_mu);
    pthread_mutex_unlock(&g_mu);
}
void mtl_drain(void) { mtl_finish(); pcache_save(); }   // going to the background: the pipelines made so far are kept for the next run

// The picture as last presented (its crop's size, and whether it is stretched over the window): app/front_android.c maps a finger on the window onto the game's touch screen
// with it (touch_map.h), the same letterbox the blit below draws. 0 until the first present
static _Atomic uint64_t g_pic_wh;
int dm_vk_picture(int *w, int *h) { uint64_t v = atomic_load(&g_pic_wh); *w = (int)(v >> 32 & 0x7fffffff); *h = (int)(uint32_t)v; return (int)(v >> 63); }
// ---------- Presentation: counted; DM_VK_SHOTS writes chosen frames as BMP files ----------
// Only the window crop is written, the part of the texture the screen shows (win_present's source rectangle): a game in handheld mode renders 1280x720 into the top left of
// a 1920x1080 swapchain texture (Harvest Moon: Echoes of Teradea, WindowCrop 0,0 1280x720), and the whole texture made a shot with the picture in one corner
static void shot(const tex_t *t, uint64_t n, const int crop[4]) {
    static char *list; static const char *dir; static int init;
    if (!init) { init = 1; const char *e = getenv("DM_VK_SHOTS"); list = e ? strdup(e) : NULL; dir = getenv("DM_VK_SHOT_DIR"); if (!dir) dir = "."; else mkdir(dir, 0775); }   // the folder may not exist yet (a fresh install's files/shots)
    if (!list) return;
    char key[24]; snprintf(key, sizeof key, ",%llu,", (unsigned long long)n); char hay[1024]; snprintf(hay, sizeof hay, ",%s,", list);
    if (!strstr(hay, key)) return;
    mtl_finish();
    lk_(&g_rmu, LW_RMU); vtex *v = tex_find(t->serial); pthread_mutex_unlock(&g_rmu);
    if (!v || (v->f.f != VK_FORMAT_R8G8B8A8_UNORM && v->f.f != VK_FORMAT_R8G8B8A8_SRGB && v->f.f != VK_FORMAT_B8G8R8A8_UNORM)) { LOG("[RED] vk: frame %llu: the presented texture is not an 8-bit RGBA image (%s)", (unsigned long long)n, v ? "format" : "never drawn"); return; }
    int32_t sx = crop ? crop[0] : 0, sy = crop ? crop[1] : 0, cw = crop && crop[2] ? crop[2] : (int32_t)v->w, ch = crop && crop[3] ? crop[3] : (int32_t)v->h;
    if (sx < 0 || sy < 0 || cw <= 0 || ch <= 0 || sx + cw > (int32_t)v->w || sy + ch > (int32_t)v->h) { sx = sy = 0; cw = (int32_t)v->w; ch = (int32_t)v->h; }   // a crop outside the texture: all of it
    uint32_t w = (uint32_t)cw, h = (uint32_t)ch; VkDeviceMemory m; void *px;
    VkBuffer b = buf_new((VkDeviceSize)w * h * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &m, &px);
    VkCommandBuffer cb = once_begin(); barrier_all(cb);
    VkBufferImageCopy c = {0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {sx, sy, 0}, {w, h, 1}};
    vkCmdCopyImageToBuffer(cb, v->img, VK_IMAGE_LAYOUT_GENERAL, b, 1, &c); once_end(cb);
    char path[1024]; snprintf(path, sizeof path, "%s/frame%05llu.bmp", dir, (unsigned long long)n);
    FILE *f = fopen(path, "wb");
    if (f) {   // 32-bit BMP, bottom-up rows, BGRA
        uint32_t sz = 54 + w * h * 4; uint8_t hd[54] = {'B', 'M'}; memcpy(hd + 2, &sz, 4); uint32_t o = 54, hs = 40; memcpy(hd + 10, &o, 4); memcpy(hd + 14, &hs, 4);
        memcpy(hd + 18, &w, 4); memcpy(hd + 22, &h, 4); hd[26] = 1; hd[28] = 32;
        fwrite(hd, 1, 54, f);
        uint8_t *row = malloc((size_t)w * 4);
        for (uint32_t y = h; y-- > 0;) {
            const uint8_t *s = (const uint8_t *)px + (size_t)y * w * 4;
            for (uint32_t x = 0; x < w; x++) { int bgr = v->f.f == VK_FORMAT_B8G8R8A8_UNORM; row[4 * x] = s[4 * x + (bgr ? 0 : 2)]; row[4 * x + 1] = s[4 * x + 1]; row[4 * x + 2] = s[4 * x + (bgr ? 2 : 0)]; row[4 * x + 3] = 255; }
            fwrite(row, 1, (size_t)w * 4, f);
        }
        free(row); fclose(f); LOG("vk: frame %llu written to %s (%ux%u; %llu draws so far, %llu dropped)", (unsigned long long)n, path, w, h, (unsigned long long)g_stat_draws, (unsigned long long)g_stat_skip);
    } else LOG("[RED] vk: frame %llu: cannot write %s: %s", (unsigned long long)n, path, strerror(errno));
    vkDestroyBuffer(g_dev, b, NULL); vkFreeMemory(g_dev, m, NULL);
}
// Diagnostics: DM_VK_RTDUMP=<frame> writes every color image the GPU rendered into, as it stands at that frame's present, to DM_VK_SHOT_DIR as
// rt_<serial>_<w>x<h>_<VkFormat>.raw (level 0, layer 0, tightly packed), for comparing a frame's passes with the Mac's; DM_VK_RTDUMP_DEC=1 adds the textures decoded
// from BC6H on upload. DM_VK_PASSLOG=<frame> logs that frame's passes (targets, clears) and draws (shaders, depth / stencil / blend state)
static int rt_bpp(VkFormat f) {
    switch (f) {
        case VK_FORMAT_R8_UNORM: return 1; case VK_FORMAT_R8G8_UNORM: case VK_FORMAT_R16_SFLOAT: case VK_FORMAT_R16_UNORM: return 2;
        case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SRGB: case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_B8G8R8A8_SRGB: case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32: case VK_FORMAT_R16G16_SFLOAT: case VK_FORMAT_R16G16_UNORM: case VK_FORMAT_R16G16_SNORM: case VK_FORMAT_R32_SFLOAT: case VK_FORMAT_R8G8B8A8_SNORM: return 4;
        case VK_FORMAT_R16G16B16A16_SFLOAT: case VK_FORMAT_R32G32_SFLOAT: return 8; case VK_FORMAT_R32G32B32A32_SFLOAT: return 16;
        default: return 0;
    }
}
static void rt_dump(uint64_t n, int force) {
    static int64_t at = -2, cnt = 1; if (at == -2) { const char *e = getenv("DM_VK_RTDUMP"); at = e ? atoll(e) : -1; const char *c = e ? strchr(e, ',') : NULL; if (c) cnt = atoll(c + 1); }   // <frame>[,<frames>]
    if (!force && ((int64_t)n < at || (int64_t)n >= at + cnt)) return;
    const char *base = getenv("DM_VK_SHOT_DIR"); if (!base) base = ".";
    char dir[512]; if (cnt > 1) { snprintf(dir, sizeof dir, "%s/f%llu", base, (unsigned long long)n); mkdir(dir, 0777); chmod(dir, 0777); } else snprintf(dir, sizeof dir, "%s", base);
    mtl_finish(); int nd = 0;
    lk_(&g_rmu, LW_RMU);
    for (int hsh = 0; hsh < NHASH; hsh++) for (vtex *v = g_tex[hsh]; v; v = v->next) {
        int bpp = rt_bpp(v->f.f); if (!v->img) continue; if (!(v->gpu || (getenv("DM_VK_RTDUMP_DEC") && v->f.dec >= DEC_BC4)) || !bpp || !(v->aspect & VK_IMAGE_ASPECT_COLOR_BIT)) continue;
        VkDeviceMemory m; void *px; VkDeviceSize sz = (VkDeviceSize)v->w * v->h * bpp;
        VkBuffer b = buf_new(sz, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &m, &px);
        VkCommandBuffer cb = once_begin(); barrier_all(cb);
        VkBufferImageCopy c = {0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {v->w, v->h, 1}};
        vkCmdCopyImageToBuffer(cb, v->img, VK_IMAGE_LAYOUT_GENERAL, b, 1, &c); once_end(cb);
        char path[1024]; snprintf(path, sizeof path, "%s/rt_%llu_%ux%u_%d.raw", dir, (unsigned long long)v->serial, v->w, v->h, (int)v->f.f);
        FILE *f = fopen(path, "wb"); if (f) { fwrite(px, 1, (size_t)sz, f); fclose(f); chmod(path, 0644); nd++; }   // readable by adb (an app's files are 0600)
        vkDestroyBuffer(g_dev, b, NULL); vkFreeMemory(g_dev, m, NULL);
    }
    pthread_mutex_unlock(&g_rmu);
    LOG("vk: frame %llu: %d render targets written to %s", (unsigned long long)n, nd, dir);
}
// ---------- Window ----------
__attribute__((weak)) void *front_native_window(int *gen) { (void)gen; return NULL; }   // app/front_android.c; none in the adb front end
__attribute__((weak)) void front_window_lock(int on) { (void)on; }
// pre: the transform the swapchain was made with (a 180-degree one when the tablet is held upside down: the picture is turned in the final blit and the overlay's rectangles, so
// the compositor need not turn it); tf/cext: the surface's current transform and extent then, to tell a real change from a driver that keeps saying SUBOPTIMAL
static struct { VkSurfaceKHR surf; VkSwapchainKHR sc; VkImage img[8]; VkImageView view[8]; uint32_t n; VkFormat fmt; VkExtent2D ext, cext; int gen;
                VkSurfaceTransformFlagBitsKHR pre, tf; uint64_t made_ns, chk_ns, nsub; const char *why; int srgb; } W;
// The swapchain's colour encoding follows the image the game presents. A blit decodes an sRGB source to linear values and stores them as they are in a UNORM destination,
// so a game whose last target is sRGB (Harvest Moon: Echoes of Teradea, Unity URP: RGBA8_SRGB) came out dark on a UNORM swapchain: every value v as (v/255)^2.4, the dim
// grass and ground near black (measured: Mac 47,65,28 → tablet 9,14,4; the decode predicts 7,13,3), bright sky and UI hardly changed. FINAL FANTASY RESONANCE presents UNORM
static int g_want_srgb;   // the presented image's encoding: the next swapchain is made with it
static int fmt_srgb(VkFormat f) { return f == VK_FORMAT_R8G8B8A8_SRGB || f == VK_FORMAT_B8G8R8A8_SRGB || f == VK_FORMAT_A8B8G8R8_SRGB_PACK32; }
#define NPRES 3
static struct { VkCommandPool pool; VkCommandBuffer cb; VkFence fence; VkSemaphore acq, done; VkFramebuffer fb; } g_pres[NPRES]; static int g_pi;
static const char *tf_name(VkSurfaceTransformFlagBitsKHR t) {
    switch (t) { case VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR: return "0"; case VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR: return "90"; case VK_SURFACE_TRANSFORM_ROTATE_180_BIT_KHR: return "180";
                 case VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR: return "270"; default: return "mirrored"; }
}
static void win_drop(void) {
    // the queue is externally synchronized: no submission while the device drains
    if (W.sc) { lk_(&g_qmu, LW_QMU); vkDeviceWaitIdle(g_dev); pthread_mutex_unlock(&g_qmu); for (uint32_t i = 0; i < W.n; i++) vkDestroyImageView(g_dev, W.view[i], NULL); vkDestroySwapchainKHR(g_dev, W.sc, NULL); W.sc = VK_NULL_HANDLE; }
    if (W.surf) { vkDestroySurfaceKHR(g_inst, W.surf, NULL); W.surf = VK_NULL_HANDLE; }
}
// DM_VK_PRESENT_PACE (vk19.63, plan v4 T2, off by default; read at start only). Nothing asked Android for a refresh rate and the swapchain was MAILBOX: the panel (modes
// 30/48/50/60/90/120/144 Hz on the Xiaomi Pad 8) could run at 120/144 Hz while the game shows 60, so frames sat 2 or 3 refreshes (judder) and the display and compositor
// ran at the high rate. With it on: the first swapchain asks for the frame cap (DM_FPS: 60, or 30 for the launcher's 30 fps profile) with
// ANativeWindow_setFrameRateWithChangeStrategy(cap, FIXED_SOURCE, ALWAYS) -- once, so the panel never changes mode during play; a later swapchain (a new window after a
// resume) asks again only if seamless (ONLY_IF_SEAMLESS) -- and the swapchain is FIFO when the cap is at most the panel's rate (read by the front end's activity thread,
// dm_front_panel_mhz), so presents follow the panel's vsync instead of replacing each other. The host's pacing already sleeps (clock_nanosleep / usleep), it never spins.
// The rate granted is in the heat-and-energy line (dm-stats asks the Display each window); displayed-frame p95/p99 in the present-to-present line
__attribute__((weak)) int dm_front_panel_mhz(void) { return 0; }
// vk19.63c: g_pace_fifo -- the swapchain is FIFO because of the switch (the host then paces 1/32 short of the period so that FIFO's vsync sets the pace, one strict pacer;
// at most minImageCount images, so the queue holds no extra frames of latency; acquire waits at most 100 ms). g_pace_low -- dm-stats saw the rate reported below the cap
// (HyperOS's adaptive 48/50 Hz): the present thread makes the swapchain again as MAILBOX, for good
static _Atomic int g_pace_fifo, g_pace_low;
int mtl_pace_fifo(void) { return atomic_load_explicit(&g_pace_fifo, memory_order_relaxed); }   // hle_nvn.c (its pacing period) and the DM_FPS=30 sleep
void vk_pace_rate_reported(int mhz) {   // dm-stats, once a window (vk_stats.c)
    if (!mhz || !atomic_load(&g_pace_fifo) || atomic_load(&g_pace_low)) return;
    int cap = mtl_fps_target();
    static int nlow;   // two windows in a row (vk19.64c): a dip on a loading screen does not turn FIFO off for the session
    if (mhz >= cap * 1000 - 500) { nlow = 0; return; }
    if (++nlow < 2) { LOG("vk: DM_VK_PRESENT_PACE: the panel's rate reported %.2f Hz is below the %d fps cap (one window; mailbox after two in a row)", mhz / 1000.0, cap); return; }
    atomic_store(&g_pace_low, 1); LOG("[RED] vk: DM_VK_PRESENT_PACE: the panel's rate reported %.2f Hz is below the %d fps cap for two windows: the swapchain goes back to mailbox", mhz / 1000.0, cap);
}
static void present_pace(ANativeWindow *aw, VkPresentModeKHR *mode) {
    tog_init(); atomic_store(&g_pace_fifo, 0); if (!atomic_load(&g_tog[TG_PACE].val)) return;
    int cap = mtl_fps_target(), mhz = dm_front_panel_mhz();
    static ANativeWindow *asked; static int nreq;
    if (aw != asked) {
        typedef int32_t (*sfr2_t)(ANativeWindow *, float, int8_t, int8_t); typedef int32_t (*sfr_t)(ANativeWindow *, float, int8_t);
        void *h = dlopen("libnativewindow.so", RTLD_NOW); sfr2_t f2 = h ? (sfr2_t)dlsym(h, "ANativeWindow_setFrameRateWithChangeStrategy") : NULL; sfr_t f1 = h ? (sfr_t)dlsym(h, "ANativeWindow_setFrameRate") : NULL;
        int8_t strategy = nreq ? 0 : 1;   // ONLY_IF_SEAMLESS after the first request, ALWAYS for the first
        int32_t r = f2 ? f2(aw, (float)cap, 1 /* FIXED_SOURCE */, strategy) : (!nreq && f1) ? f1(aw, (float)cap, 1) : -1;
        LOG("vk: DM_VK_PRESENT_PACE: %d Hz asked of the window (%s, fixed source, %s): %d; the panel %.2f Hz now", cap, f2 ? "setFrameRateWithChangeStrategy" : f1 ? "setFrameRate" : "no setFrameRate",
            strategy ? "change always" : "only if seamless", (int)r, mhz / 1000.0);
        asked = aw; nreq++;
    }
    if (mhz > 0 && cap * 1000 <= mhz + 500 && !atomic_load(&g_pace_low)) { *mode = VK_PRESENT_MODE_FIFO_KHR; atomic_store(&g_pace_fifo, 1); }
    LOG("vk: DM_VK_PRESENT_PACE: %s (cap %d fps, panel %.2f Hz reported%s)", atomic_load(&g_pace_fifo) ? "fifo" : "mailbox", cap, mhz / 1000.0,
        atomic_load(&g_pace_low) ? ": below the cap, mailbox" : mhz ? "" : ": unknown, mailbox kept");
}
static int win_make(ANativeWindow *aw) {
    VkAndroidSurfaceCreateInfoKHR si = {VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR, NULL, 0, aw};
    if (vkCreateAndroidSurfaceKHR(g_inst, &si, NULL, &W.surf) != VK_SUCCESS) { RED_ONCE(90, "no Vulkan surface for the window"); return 0; }
    VkBool32 ok = 0; vkGetPhysicalDeviceSurfaceSupportKHR(g_pd, g_qf, W.surf, &ok); if (!ok) { RED_ONCE(91, "the queue cannot present to the window"); return 0; }
    VkSurfaceCapabilitiesKHR cap; vkGetPhysicalDeviceSurfaceCapabilitiesKHR(g_pd, W.surf, &cap);
    uint32_t nf = 0; vkGetPhysicalDeviceSurfaceFormatsKHR(g_pd, W.surf, &nf, NULL); VkSurfaceFormatKHR sf[32]; if (nf > 32) nf = 32; vkGetPhysicalDeviceSurfaceFormatsKHR(g_pd, W.surf, &nf, sf);
    VkSurfaceFormatKHR pick = sf[0]; for (uint32_t i = 0; i < nf; i++) if (sf[i].format == VK_FORMAT_R8G8B8A8_UNORM || sf[i].format == VK_FORMAT_B8G8R8A8_UNORM) { pick = sf[i]; break; }
    if (g_want_srgb) { for (uint32_t i = 0; i < nf; i++) if (sf[i].format == VK_FORMAT_R8G8B8A8_SRGB || sf[i].format == VK_FORMAT_B8G8R8A8_SRGB) { pick = sf[i]; break; }   // an sRGB image presented: see g_want_srgb
                       if (!fmt_srgb(pick.format)) RED_ONCE(111, "the game presents an sRGB image and the window offers no sRGB swapchain format: the picture is too dark"); }
    W.srgb = fmt_srgb(pick.format);
    uint32_t npm = 0; vkGetPhysicalDeviceSurfacePresentModesKHR(g_pd, W.surf, &npm, NULL); VkPresentModeKHR pm[8]; if (npm > 8) npm = 8; vkGetPhysicalDeviceSurfacePresentModesKHR(g_pd, W.surf, &npm, pm);
    VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR; for (uint32_t i = 0; i < npm; i++) if (pm[i] == VK_PRESENT_MODE_MAILBOX_KHR) mode = pm[i];   // the host paces presents itself; mailbox never blocks it
    present_pace(aw, &mode);   // DM_VK_PRESENT_PACE (vk19.63): the panel's rate requested for the cap, FIFO when the cap fits it
    // The swapchain is made the window's way up and Android's compositor turns it (a phone whose screen is portrait by nature, playing in landscape, reports its
    // current extent and transform rotated); the window's own size is that way up
    // Upright or upside down the swapchain is made the screen's current way (180 degrees is turned by us, for free in the blit); a quarter turn is left to the compositor
    VkSurfaceTransformFlagBitsKHR pre = cap.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR ? VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR : cap.currentTransform;
    if (cap.currentTransform == VK_SURFACE_TRANSFORM_ROTATE_180_BIT_KHR && (cap.supportedTransforms & VK_SURFACE_TRANSFORM_ROTATE_180_BIT_KHR)) pre = VK_SURFACE_TRANSFORM_ROTATE_180_BIT_KHR;
    W.pre = pre; W.tf = cap.currentTransform; W.cext = cap.currentExtent;
    W.ext = (VkExtent2D){(uint32_t)ANativeWindow_getWidth(aw), (uint32_t)ANativeWindow_getHeight(aw)};
    if (pre == cap.currentTransform && cap.currentExtent.width != UINT32_MAX) W.ext = cap.currentExtent;
    if (W.ext.width < cap.minImageExtent.width) W.ext.width = cap.minImageExtent.width; if (W.ext.width > cap.maxImageExtent.width) W.ext.width = cap.maxImageExtent.width;
    if (W.ext.height < cap.minImageExtent.height) W.ext.height = cap.minImageExtent.height; if (W.ext.height > cap.maxImageExtent.height) W.ext.height = cap.maxImageExtent.height;
    uint32_t nimg = cap.minImageCount + 1; if (cap.maxImageCount && nimg > cap.maxImageCount) nimg = cap.maxImageCount;
    if (atomic_load(&g_pace_fifo)) {   // DM_VK_PRESENT_PACE's FIFO: no spare image beyond minImageCount (a full queue blocks instead of adding latency), but at least 3 (vk19.64c):
        // with two, FIFO locks a 45-58 fps game to every other vsync (~30 fps)
        nimg = cap.minImageCount > 3 ? cap.minImageCount : 3; if (cap.maxImageCount && nimg > cap.maxImageCount) nimg = cap.maxImageCount;
        LOG("vk: DM_VK_PRESENT_PACE: fifo with %u images (minImageCount %u, maxImageCount %u)", nimg, cap.minImageCount, cap.maxImageCount); }
    VkCompositeAlphaFlagBitsKHR ca = cap.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR ? VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR : VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
    VkSwapchainCreateInfoKHR ci = {VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR, NULL, 0, W.surf, nimg, pick.format, pick.colorSpace, W.ext, 1,
                                   VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, pre, ca, mode, VK_TRUE, VK_NULL_HANDLE};
    if (vkCreateSwapchainKHR(g_dev, &ci, NULL, &W.sc) != VK_SUCCESS) { RED_ONCE(92, "no swapchain for the window"); return 0; }
    W.fmt = pick.format; W.n = 8; vkGetSwapchainImagesKHR(g_dev, W.sc, &W.n, W.img);
    for (uint32_t i = 0; i < W.n; i++) { VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, W.img[i], VK_IMAGE_VIEW_TYPE_2D, W.fmt, {0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
                                         VKCHECK(vkCreateImageView(g_dev, &vi, NULL, &W.view[i])); }
    if (!g_pres[0].pool) for (int i = 0; i < NPRES; i++) {
        VkCommandPoolCreateInfo pi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, g_qf}; VKCHECK(vkCreateCommandPool(g_dev, &pi, NULL, &g_pres[i].pool));
        VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, g_pres[i].pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1}; VKCHECK(vkAllocateCommandBuffers(g_dev, &ai, &g_pres[i].cb));
        VkFenceCreateInfo fi = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, NULL, VK_FENCE_CREATE_SIGNALED_BIT}; VKCHECK(vkCreateFence(g_dev, &fi, NULL, &g_pres[i].fence));
        VkSemaphoreCreateInfo sci = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO}; VKCHECK(vkCreateSemaphore(g_dev, &sci, NULL, &g_pres[i].acq)); VKCHECK(vkCreateSemaphore(g_dev, &sci, NULL, &g_pres[i].done));
    }
    LOG("vk: swapchain %ux%u, %u images, format %d, %s, transform %s (screen %s), why: %s", W.ext.width, W.ext.height, W.n, W.fmt, mode == VK_PRESENT_MODE_MAILBOX_KHR ? "mailbox" : "fifo",
        tf_name(pre), tf_name(cap.currentTransform), W.why ? W.why : "new window");
    W.why = NULL; W.made_ns = now_ns(); W.nsub = 0;
    return 1;
}
// A present or acquire said SUBOPTIMAL. Android says it every frame while the swapchain's transform is not the screen's (a quarter turn, which the compositor does), so
// making a new swapchain then only shows black frames, over and over. A new one is made only when the screen's transform or size really changed since this one was made,
// and at most twice a second
static void win_suboptimal(void) {
    W.nsub++;
    uint64_t t = now_ns(); if (t - W.made_ns < 500000000ull || t - W.chk_ns < 500000000ull) return;
    W.chk_ns = t; VkSurfaceCapabilitiesKHR cap; if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(g_pd, W.surf, &cap) != VK_SUCCESS) return;
    if (cap.currentTransform != W.tf) W.why = "the screen turned";
    else if (cap.currentExtent.width != W.cext.width || cap.currentExtent.height != W.cext.height) W.why = "the window's size changed";
    else return;
    LOG("vk: suboptimal %llu times; %s (transform %s -> %s, %ux%u -> %ux%u)", (unsigned long long)W.nsub, W.why, tf_name(W.tf), tf_name(cap.currentTransform),
        W.cext.width, W.cext.height, cap.currentExtent.width, cap.currentExtent.height);
    W.gen = -1;
}
// The overlay's rectangles in window coordinates, turned with the swapchain
static void ovl_clear(VkCommandBuffer cb, const VkClearAttachment *a, uint32_t n, VkClearRect *r) {
    if (W.pre == VK_SURFACE_TRANSFORM_ROTATE_180_BIT_KHR)
        for (uint32_t i = 0; i < n; i++) { r[i].rect.offset.x = (int32_t)W.ext.width - r[i].rect.offset.x - (int32_t)r[i].rect.extent.width;
                                           r[i].rect.offset.y = (int32_t)W.ext.height - r[i].rect.offset.y - (int32_t)r[i].rect.extent.height; }
    vkCmdClearAttachments(cb, 1, a, n, r);
}
// Over the picture: the frame-rate counter (cleared rectangles, a 3x5 font) and the on-screen controls (front_overlay): sprites from the controls' atlas, textured quads
// alpha-blended over the picture (ovl_sprites), brighter while pressed; without an atlas (the adb program, an APK without one) button outlines and labels as before
typedef struct { float x, y, w, h; int on; char label[3]; int sprite; float alpha; } dm_ovl;   // sprite: cell of the atlas, -1 none; alpha: the layout's opacity (app/front_android.c)
__attribute__((weak)) int front_overlay(dm_ovl *out, int max) { (void)out; (void)max; return 0; }
// The atlas (app/front_android.c: the APK's assets/controls/atlas.png, decoded premultiplied RGBA; dm/android/tools/controls_atlas.py makes it from the SVG masters in app/res_svg)
__attribute__((weak)) int front_overlay_atlas(const void **rgba, int *w, int *h, int *cell, int *cols) { (void)rgba; (void)w; (void)h; (void)cell; (void)cols; return 0; }
__attribute__((weak)) void front_overlay_atlas_release(void) {}
static const uint32_t OVL_VS[] =
#include "ovl_vert.inc"
;
static const uint32_t OVL_FS[] =
#include "ovl_frag.inc"
;
static struct { int state; VkImage img; VkDeviceMemory mem; VkImageView view; VkSampler samp; VkDescriptorSetLayout dsl; VkDescriptorPool dp; VkDescriptorSet ds; VkPipelineLayout pl;
                VkPipeline pipe; VkFormat pfmt; int w, h, cell, cols; } G_;   // state: 0 not tried, 1 ready, -1 no atlas
typedef struct { float rect[4], uv[4], tint[4], scr[4]; } ovl_pc;   // ovl.vert / ovl.frag push constants
static void ovl_layout(VkCommandBuffer cb, VkImage img, VkImageLayout from, VkImageLayout to, VkAccessFlags sa, VkAccessFlags da, VkPipelineStageFlags ss, VkPipelineStageFlags ds) {
    VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, sa, da, from, to, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, img, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    vkCmdPipelineBarrier(cb, ss, ds, 0, 0, NULL, 0, NULL, 1, &b);
}
static int ovl_init(void) {   // the atlas as a sampled image, once (the first frame with on-screen controls)
    if (G_.state) return G_.state > 0;
    G_.state = -1;
    const void *px = NULL; int w = 0, h = 0, cell = 0, cols = 0;
    if (!front_overlay_atlas(&px, &w, &h, &cell, &cols) || !px || w <= 0 || h <= 0 || cell <= 0 || cols <= 0) { LOG("vk: on-screen controls: no sprite atlas, drawn as outlines"); return 0; }
    VkImageCreateInfo ii = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, {(uint32_t)w, (uint32_t)h, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT,
                            VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
    if (vkCreateImage(g_dev, &ii, NULL, &G_.img) != VK_SUCCESS) { front_overlay_atlas_release(); return 0; }
    VkMemoryRequirements mr; vkGetImageMemoryRequirements(g_dev, G_.img, &mr); VkDeviceSize off; img_mem(mr, &G_.mem, &off); VKCHECK(vkBindImageMemory(g_dev, G_.img, G_.mem, off));
    VkDeviceSize n = (VkDeviceSize)w * h * 4; VkDeviceMemory sm; void *map; VkBuffer sb = buf_new(n, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &sm, &map); memcpy(map, px, n);
    front_overlay_atlas_release();
    VkCommandBuffer cb = once_begin();
    ovl_layout(cb, G_.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkBufferImageCopy bc = {0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {(uint32_t)w, (uint32_t)h, 1}};
    vkCmdCopyBufferToImage(cb, sb, G_.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bc);
    ovl_layout(cb, G_.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
               VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    once_end(cb); vkDestroyBuffer(g_dev, sb, NULL); vkFreeMemory(g_dev, sm, NULL);
    VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, G_.img, VK_IMAGE_VIEW_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, {0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VKCHECK(vkCreateImageView(g_dev, &vi, NULL, &G_.view));
    VkSamplerCreateInfo si = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, NULL, 0, VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 0, VK_FALSE, 1, VK_FALSE, VK_COMPARE_OP_ALWAYS, 0, 0, VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK, VK_FALSE};
    VKCHECK(vkCreateSampler(g_dev, &si, NULL, &G_.samp));
    VkDescriptorSetLayoutBinding lb = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL};
    VkDescriptorSetLayoutCreateInfo li = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 1, &lb}; VKCHECK(vkCreateDescriptorSetLayout(g_dev, &li, NULL, &G_.dsl));
    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1}; VkDescriptorPoolCreateInfo pi = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, 1, 1, &ps};
    VKCHECK(vkCreateDescriptorPool(g_dev, &pi, NULL, &G_.dp));
    VkDescriptorSetAllocateInfo ai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, G_.dp, 1, &G_.dsl}; VKCHECK(vkAllocateDescriptorSets(g_dev, &ai, &G_.ds));
    VkDescriptorImageInfo di = {G_.samp, G_.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet wd = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, G_.ds, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &di, NULL, NULL}; vkUpdateDescriptorSets(g_dev, 1, &wd, 0, NULL);
    VkPushConstantRange pr = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(ovl_pc)};
    VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &G_.dsl, 1, &pr}; VKCHECK(vkCreatePipelineLayout(g_dev, &pli, NULL, &G_.pl));
    G_.w = w; G_.h = h; G_.cell = cell; G_.cols = cols; G_.state = 1;
    LOG("vk: on-screen controls: sprite atlas %dx%d (cells of %d) ready", w, h, cell);
    return 1;
}
static int ovl_pipe(VkFormat fmt) {   // the sprites' pipeline for the swapchain's format (made again only if a new swapchain has another one; the old one is left, as frames in flight may use it)
    if (G_.pipe && G_.pfmt == fmt) return 1;
    VkShaderModule m[2]; VkShaderModuleCreateInfo mi[2] = {{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sizeof OVL_VS, OVL_VS}, {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sizeof OVL_FS, OVL_FS}};
    VKCHECK(vkCreateShaderModule(g_dev, &mi[0], NULL, &m[0])); VKCHECK(vkCreateShaderModule(g_dev, &mi[1], NULL, &m[1]));
    VkPipelineShaderStageCreateInfo st[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_VERTEX_BIT, m[0], "main", NULL},
                                             {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_FRAGMENT_BIT, m[1], "main", NULL}};
    int32_t srgb_out = fmt_srgb(fmt); VkSpecializationMapEntry sme = {0, 0, sizeof srgb_out}; VkSpecializationInfo spi = {1, &sme, sizeof srgb_out, &srgb_out};
    st[1].pSpecializationInfo = &spi;   // ovl.frag SRGB_OUT: the sprites keep their look on an sRGB swapchain
    VkPipelineVertexInputStateCreateInfo vi = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, NULL, 0, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP, VK_FALSE};
    VkPipelineViewportStateCreateInfo vp = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, NULL, 0, 1, NULL, 1, NULL};
    VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, NULL, 0, VK_FALSE, VK_FALSE, VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, VK_FALSE, 0, 0, 0, 1};
    VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, NULL, 0, VK_SAMPLE_COUNT_1_BIT};
    VkPipelineColorBlendAttachmentState ba = {VK_TRUE, VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, VK_BLEND_OP_ADD, VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, VK_BLEND_OP_ADD, 0xf};   // premultiplied
    VkPipelineColorBlendStateCreateInfo cbs = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, NULL, 0, VK_FALSE, VK_LOGIC_OP_COPY, 1, &ba, {0}};
    VkDynamicState dy[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR}; VkPipelineDynamicStateCreateInfo ds = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, NULL, 0, 2, dy};
    VkPipelineRenderingCreateInfo prc = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, NULL, 0, 1, &fmt, VK_FORMAT_UNDEFINED, VK_FORMAT_UNDEFINED};
    VkAttachmentLoadOp ld = VK_ATTACHMENT_LOAD_OP_LOAD;
    VkGraphicsPipelineCreateInfo ci = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, g_dynrend ? &prc : NULL, 0, 2, st, &vi, &ia, NULL, &vp, &rs, &ms, NULL, &cbs, &ds, G_.pl,
                                       g_dynrend ? VK_NULL_HANDLE : rp_get(&fmt, &ld, 1, VK_FORMAT_UNDEFINED, 0, ld, ld, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL), 0, VK_NULL_HANDLE, -1};
    VkPipeline p; VkResult r = vkCreateGraphicsPipelines(g_dev, g_pcache, 1, &ci, NULL, &p);
    vkDestroyShaderModule(g_dev, m[0], NULL); vkDestroyShaderModule(g_dev, m[1], NULL);
    if (r != VK_SUCCESS) { RED_ONCE(94, "the on-screen controls' pipeline was not created (%d): drawn as outlines", (int)r); G_.state = -1; return 0; }
    G_.pipe = p; G_.pfmt = fmt; return 1;
}
static void ovl_sprites(VkCommandBuffer cb, const dm_ovl *ov, int nov) {   // inside the overlay's pass: each sprite item a quad
    VkViewport v = {0, 0, (float)W.ext.width, (float)W.ext.height, 0, 1}; VkRect2D sc = {{0, 0}, W.ext};
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, G_.pipe); vkCmdSetViewport(cb, 0, 1, &v); vkCmdSetScissor(cb, 0, 1, &sc);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, G_.pl, 0, 1, &G_.ds, 0, NULL);
    float iw = 1.0f / (float)W.ext.width, ih = 1.0f / (float)W.ext.height, rot = W.pre == VK_SURFACE_TRANSFORM_ROTATE_180_BIT_KHR;
    for (int i = 0; i < nov; i++) {
        const dm_ovl *o = &ov[i]; if (o->sprite < 0 || o->alpha <= 0) continue;
        int col = o->sprite % G_.cols, row = o->sprite / G_.cols;
        ovl_pc pc = {{o->x, o->y, o->w, o->h}, {((float)(col * G_.cell) + 0.5f) / (float)G_.w, ((float)(row * G_.cell) + 0.5f) / (float)G_.h, (float)(G_.cell - 1) / (float)G_.w, (float)(G_.cell - 1) / (float)G_.h},
                     {o->on ? 1.25f : 1.0f, o->on ? 0.12f : 0.0f, o->alpha, 0}, {iw, ih, rot, 0}};
        vkCmdPushConstants(cb, G_.pl, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof pc, &pc);
        vkCmdDraw(cb, 4, 1, 0, 0);
    }
}
static double g_fps_shown;
static uint16_t glyph(char c) {   // rows top to bottom, 3 bits each
    static const char *const G[] = {"0111101101101111", "1010010010010010", "2111001111100111", "3111001111001111", "4101101111001001", "5111100111001111", "6111100111101111",
                                    "7111001001001001", "8111101111101111", "9111101111001111", "A010101111101101", "B110101110101110", "X101101010101101", "Y101101010010010",
                                    "L100100100100111", "R110101110101101", "Z111001010100111", "+000010111010000", "-000000111000000",
                                    "M101111111101101", "E111100111100111", "N110101101101101", "I111010010010111", "P110101110100100", "K101101110101101", "S111100111001111",
                                    "H101101111101101", "D110101101101110", "T111010010010010", "U101101101101111", "W101101111111101", "/001001010100100", ".000000000000010",
                                    "J001001001101111"};
    for (size_t i = 0; i < sizeof G / sizeof *G; i++) if (G[i][0] == c) { uint16_t g = 0; for (int k = 1; k <= 15; k++) g = (uint16_t)(g << 1 | (G[i][k] == '1')); return g; }
    return 0;
}
static uint32_t text_rects(VkClearRect *r, uint32_t nr, uint32_t max, const char *t, int32_t x0, int32_t y0, uint32_t cell) {
    for (size_t d = 0; t[d] && nr + 15 <= max; d++) { uint16_t g = glyph(t[d]);
        for (int row = 0; row < 5; row++) for (int c = 0; c < 3; c++) if (g >> (14 - 3 * row - c) & 1)
            r[nr++] = (VkClearRect){{{x0 + (int32_t)((d * 4 + c) * cell), y0 + (int32_t)(row * cell)}, {cell, cell}}, 0, 1}; }
    return nr;
}
static VkFramebuffer overlay_draw(VkCommandBuffer cb, uint32_t img, int show_fps, const dm_ovl *ov, int nov) {
    rp_desc d = {.n = 1, .cv = {W.view[img]}, .cf = {W.fmt}, .cl = {VK_ATTACHMENT_LOAD_OP_LOAD}, .w = W.ext.width, .h = W.ext.height, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkFramebuffer fb = rp_begin(cb, &d);
    enum { MAXR = 1024 }; static VkClearRect lines[MAXR], fills[MAXR], fg[MAXR], bg[64]; uint32_t nl = 0, nf = 0, ng = 0, nb = 0;
    #define CLIP(x0, y0, w_, h_) do { int32_t X = (int32_t)(x0), Y = (int32_t)(y0), XW = X + (int32_t)(w_), YH = Y + (int32_t)(h_); if (X < 0) X = 0; if (Y < 0) Y = 0; \
        if (XW > (int32_t)W.ext.width) XW = (int32_t)W.ext.width; if (YH > (int32_t)W.ext.height) YH = (int32_t)W.ext.height; \
        if (XW > X && YH > Y) rect = (VkClearRect){{{X, Y}, {(uint32_t)(XW - X), (uint32_t)(YH - Y)}}, 0, 1}; else rect.rect.extent.width = 0; } while (0)
    VkClearRect rect;
    if (show_fps) {
        char t[8]; snprintf(t, sizeof t, "%.0f", g_fps_shown); uint32_t cell = W.ext.height / 120 ? W.ext.height / 120 : 2;
        CLIP(cell * 2, cell * 2, (strlen(t) * 4 + 1) * cell, 7 * cell); if (rect.rect.extent.width) bg[nb++] = rect;
        ng = text_rects(fg, ng, MAXR, t, (int32_t)(cell * 3), (int32_t)(cell * 3), cell);
    }
    int spr = nov && ovl_init() && ovl_pipe(W.fmt);   // sprites: their own pipeline, after the rectangles
    for (int i = 0; i < nov && nl + 4 <= MAXR; i++) {
        const dm_ovl *o = &ov[i]; float t = o->w / 18 > 2 ? o->w / 18 : 2;
        if (spr && o->sprite >= 0) continue;
        if (o->on) { CLIP(o->x, o->y, o->w, o->h); if (rect.rect.extent.width && nf < MAXR) fills[nf++] = rect; }
        CLIP(o->x, o->y, o->w, t); if (rect.rect.extent.width) lines[nl++] = rect;
        CLIP(o->x, o->y + o->h - t, o->w, t); if (rect.rect.extent.width) lines[nl++] = rect;
        CLIP(o->x, o->y, t, o->h); if (rect.rect.extent.width) lines[nl++] = rect;
        CLIP(o->x + o->w - t, o->y, t, o->h); if (rect.rect.extent.width) lines[nl++] = rect;
        if (o->label[0]) { uint32_t cell = (uint32_t)(o->h / 9) ? (uint32_t)(o->h / 9) : 1, tw = (uint32_t)strlen(o->label) * 4 * cell - cell;
                           ng = text_rects(fg, ng, MAXR, o->label, (int32_t)(o->x + (o->w - tw) / 2), (int32_t)(o->y + (o->h - 5 * cell) / 2), cell); }
    }
    #undef CLIP
    VkClearAttachment black = {VK_IMAGE_ASPECT_COLOR_BIT, 0, {.color = {{0, 0, 0, 1}}}}, white = {VK_IMAGE_ASPECT_COLOR_BIT, 0, {.color = {{0.85f, 0.85f, 0.9f, 1}}}},
                      grey = {VK_IMAGE_ASPECT_COLOR_BIT, 0, {.color = {{0.35f, 0.35f, 0.45f, 1}}}}, green = {VK_IMAGE_ASPECT_COLOR_BIT, 0, {.color = {{0.3f, 1, 0.3f, 1}}}};
    if (W.srgb) {   // an sRGB swapchain encodes clear values: the display values decoded first, so the counter keeps its colours
        VkClearAttachment *ca[] = {&white, &grey, &green};
        for (int i = 0; i < 3; i++) for (int k = 0; k < 3; k++) ca[i]->clearValue.color.float32[k] = powf(ca[i]->clearValue.color.float32[k], 2.2f); }
    if (nb) ovl_clear(cb, &black, nb, bg);
    if (nf) ovl_clear(cb, &grey, nf, fills);
    if (nl) ovl_clear(cb, &white, nl, lines);
    if (ng) ovl_clear(cb, nov ? &white : &green, ng, fg);
    if (spr) ovl_sprites(cb, ov, nov);
    rp_end(cb);
    return fb;
}
static void img_barrier(VkCommandBuffer cb, VkImage img, VkImageLayout from, VkImageLayout to) {
    VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, from, to,
                              VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, img, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 1, &b);
}
// The window is going away (app/front_android.c, NativeActivity onNativeWindowDestroyed, holding front_window_lock): the swapchain and surface go now, before Android
// takes the window back; the device, pipelines and every game image stay. The next present after a new window makes a new swapchain (the generation differs)
void dm_vk_window_lost(void) { if (!g_dev) return; win_drop(); W.gen = -1; LOG("vk: window gone: swapchain and surface released"); }
static void win_present(const tex_t *t, const int crop[4]) {
    int gen; ANativeWindow *aw = front_native_window(&gen);
    front_window_lock(1);
    aw = front_native_window(&gen);
    if (aw && W.sc && atomic_load(&g_pace_fifo) && atomic_load(&g_pace_low)) { W.gen = -1; W.why = "panel rate below the cap (DM_VK_PRESENT_PACE)"; }   // back to mailbox
    {   // the swapchain's encoding follows the presented image's (g_want_srgb); a change makes a new swapchain
        lk_(&g_rmu, LW_RMU); vtex *pv = tex_find(t->serial); int s_ = pv ? fmt_srgb(pv->f.f) : g_want_srgb; pthread_mutex_unlock(&g_rmu);
        if (s_ != g_want_srgb) { g_want_srgb = s_; if (W.sc) { W.gen = -1; W.why = s_ ? "the game presents an sRGB image" : "the game presents a UNORM image"; } }
    }
    if (!aw || gen != W.gen) { win_drop(); W.gen = gen; if (aw) win_make(aw); }
    if (!aw || !W.sc) { front_window_lock(0); return; }
    lk_(&g_rmu, LW_RMU); vtex *v = tex_find(t->serial); pthread_mutex_unlock(&g_rmu);
    if (!v) { front_window_lock(0); return; }
    typeof(g_pres[0]) *P_ = &g_pres[g_pi]; g_pi = (g_pi + 1) % NPRES;
    { uint64_t w0 = now_ns(); vkWaitForFences(g_dev, 1, &P_->fence, VK_TRUE, UINT64_MAX); atomic_fetch_add(&g_pwait_ns, now_ns() - w0); }
    if (P_->fb) { vkDestroyFramebuffer(g_dev, P_->fb, NULL); P_->fb = VK_NULL_HANDLE; }
    // under DM_VK_PRESENT_PACE's FIFO at most 100 ms (the window lock is held: a backgrounded window must not keep the activity thread's window-destroyed waiting a second); the frame is skipped
    uint32_t idx; VkResult r = vkAcquireNextImageKHR(g_dev, W.sc, atomic_load(&g_pace_fifo) ? 100000000ull : 1000000000ull, P_->acq, VK_NULL_HANDLE, &idx);
    if (r == VK_ERROR_OUT_OF_DATE_KHR) { W.gen = -1; W.why = "out of date at acquire"; front_window_lock(0); return; }
    if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) { front_window_lock(0); return; }
    if (r == VK_SUBOPTIMAL_KHR) win_suboptimal();
    vkResetFences(g_dev, 1, &P_->fence);
    VkCommandBuffer cb = P_->cb; vkResetCommandBuffer(cb, 0);
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL}; vkBeginCommandBuffer(cb, &bi);
    barrier_all(cb);   // the game's submissions before this one wrote the image
    img_barrier(cb, W.img[idx], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkClearColorValue black = {{0, 0, 0, 1}}; VkImageSubresourceRange rr = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(cb, W.img[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &rr);
    int32_t sx = crop ? crop[0] : 0, sy = crop ? crop[1] : 0, sw = crop && crop[2] ? crop[2] : (int32_t)v->w, sh = crop && crop[3] ? crop[3] : (int32_t)v->h;
    static int stretch = -1; if (stretch < 0) { const char *e = getenv("DM_SCREEN_FIT"); stretch = e && !strcmp(e, "stretch"); }
    atomic_store(&g_pic_wh, (uint64_t)(uint32_t)sw << 32 | (uint32_t)sh | (stretch ? 1ull << 63 : 0));   // what the front end maps fingers onto (dm_vk_picture)
    int32_t dw = (int32_t)W.ext.width, dh = (int32_t)W.ext.height, dx = 0, dy = 0;
    if (!stretch) { if ((int64_t)dw * sh > (int64_t)dh * sw) { dw = (int32_t)((int64_t)dh * sw / sh); dx = ((int32_t)W.ext.width - dw) / 2; } else { dh = (int32_t)((int64_t)dw * sh / sw); dy = ((int32_t)W.ext.height - dh) / 2; } }
    VkImageBlit b = {{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {{sx, sy, 0}, {sx + sw, sy + sh, 1}}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {{dx, dy, 0}, {dx + dw, dy + dh, 1}}};
    if (W.pre == VK_SURFACE_TRANSFORM_ROTATE_180_BIT_KHR) {   // upside down: the destination corners swapped turn the picture half round
        b.dstOffsets[0] = (VkOffset3D){(int32_t)W.ext.width - dx, (int32_t)W.ext.height - dy, 0}; b.dstOffsets[1] = (VkOffset3D){(int32_t)W.ext.width - dx - dw, (int32_t)W.ext.height - dy - dh, 1}; }
    vkCmdBlitImage(cb, v->img, VK_IMAGE_LAYOUT_GENERAL, W.img[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &b, VK_FILTER_LINEAR);
    static int show = -1; if (show < 0) { const char *e = getenv("DM_SHOW_FPS"); show = e && atoi(e); }
    static dm_ovl ov[64]; int nov = front_overlay(ov, 64);   // buttons, and a ring of dots and a knob per stick held
    if (show || nov) { img_barrier(cb, W.img[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL); P_->fb = overlay_draw(cb, idx, show, ov, nov);
                img_barrier(cb, W.img[idx], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR); }
    else img_barrier(cb, W.img[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    vkEndCommandBuffer(cb);
    VkPipelineStageFlags ws[2] = {VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT};
    VkSemaphore wsem[2] = {P_->acq, g_tl}, ssem[2] = {P_->done, g_tl}; uint64_t wv[2] = {0, 0}, sv[2] = {0, 0};
    VkTimelineSemaphoreSubmitInfo ts = {VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO, NULL, 2, wv, 2, sv};
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, g_tl ? &ts : NULL, g_tl ? 2 : 1, wsem, ws, 1, &cb, g_tl ? 2 : 1, ssem};
    VkPresentInfoKHR pi = {VK_STRUCTURE_TYPE_PRESENT_INFO_KHR, NULL, 1, &P_->done, 1, &W.sc, &idx, NULL};
    lk_(&g_qmu, LW_QMU);
    if (g_tl) { wv[1] = g_tlv; sv[1] = ++g_tlv; g_after_present = 1; }
    vkQueueSubmit(g_q, 1, &si, P_->fence); r = vkQueuePresentKHR(g_q, &pi); pthread_mutex_unlock(&g_qmu);
    if (r == VK_ERROR_OUT_OF_DATE_KHR) { W.gen = -1; W.why = "out of date at present"; } else if (r == VK_SUBOPTIMAL_KHR && W.gen != -1) win_suboptimal();
    front_window_lock(0);
}
static _Atomic uint64_t g_presents;
// ---------- Shader pre-warm before the game starts ----------
// Every pipeline earlier runs made (pipeline_keys.bin) is made again before the game boots, on worker threads, behind a screen "MENYIAPKAN SHADER x / y" with a progress bar, so
// effects and skills met before never stall the game for a first-time shader load or compile (100-300 ms each in FINAL FANTASY RESONANCE's battles). The player's own tap on the
// screen skips the wait (dm_vk_prewarm_cancel, from the front end's input); the rest is made in the background as the game starts. The first run after the list grows takes
// longer (real compiles), later ones only seconds (the driver's pipeline cache)
static _Atomic int g_pw_cancel; static pso_rec *g_pw_recs; static uint32_t g_pw_total; static _Atomic uint32_t g_pw_next, g_pw_done;
void dm_vk_prewarm_cancel(void) { atomic_store(&g_pw_cancel, 1); }
static void *pw_worker(void *a) { pthread_setname_np("dm-prewarm");
    (void)a;
    for (uint32_t i; (i = atomic_fetch_add(&g_pw_next, 1)) < g_pw_total;) { pso_precreate_one(&g_pw_recs[i]); atomic_fetch_add(&g_pw_done, 1); }
    return NULL;
}
static void screen_draw(const char *t1, const char *t2, uint32_t done, uint32_t total);
static void pw_draw(uint32_t done, uint32_t total) {   // one frame of the pre-warm screen
    char t1[64]; snprintf(t1, sizeof t1, "MENYIAPKAN SHADER %u / %u", done, total);
    screen_draw(t1, "KETUK UNTUK LEWATI", done, total);
}
// The hibernation restore's screen (app/front_android.c, before the host reads the snapshot): drawn until the game's first frame replaces it
void dm_vk_message(const char *t1, const char *t2) {
    vk_init();
    int gen; for (int i = 0; i < 40; i++) { front_window_lock(1); ANativeWindow *aw = front_native_window(&gen); front_window_lock(0); if (aw) break; usleep(50000); }
    for (int i = 0; i < 3; i++) screen_draw(t1, t2, 0, 0);
}
static void screen_draw(const char *t1, const char *t2, uint32_t done, uint32_t total) {   // one frame: t1 large, t2 small, a progress bar when total > 0
    int gen; front_window_lock(1); ANativeWindow *aw = front_native_window(&gen);
    if (!aw || gen != W.gen) { win_drop(); W.gen = gen; if (aw) win_make(aw); }
    if (!aw || !W.sc) { front_window_lock(0); return; }
    typeof(g_pres[0]) *P_ = &g_pres[g_pi]; g_pi = (g_pi + 1) % NPRES;
    vkWaitForFences(g_dev, 1, &P_->fence, VK_TRUE, UINT64_MAX);
    if (P_->fb) { vkDestroyFramebuffer(g_dev, P_->fb, NULL); P_->fb = VK_NULL_HANDLE; }
    // under DM_VK_PRESENT_PACE's FIFO at most 100 ms (the window lock is held: a backgrounded window must not keep the activity thread's window-destroyed waiting a second); the frame is skipped
    uint32_t idx; VkResult r = vkAcquireNextImageKHR(g_dev, W.sc, atomic_load(&g_pace_fifo) ? 100000000ull : 1000000000ull, P_->acq, VK_NULL_HANDLE, &idx);
    if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) { if (r == VK_ERROR_OUT_OF_DATE_KHR) { W.gen = -1; W.why = "out of date at acquire"; } front_window_lock(0); return; }
    if (r == VK_SUBOPTIMAL_KHR) win_suboptimal();
    vkResetFences(g_dev, 1, &P_->fence);
    VkCommandBuffer cb = P_->cb; vkResetCommandBuffer(cb, 0);
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL}; vkBeginCommandBuffer(cb, &bi);
    img_barrier(cb, W.img[idx], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkClearColorValue bgc = {{0.06f, 0.07f, 0.11f, 1}}; VkImageSubresourceRange rr = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(cb, W.img[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &bgc, 1, &rr);
    img_barrier(cb, W.img[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    rp_desc d = {.n = 1, .cv = {W.view[idx]}, .cf = {W.fmt}, .cl = {VK_ATTACHMENT_LOAD_OP_LOAD}, .w = W.ext.width, .h = W.ext.height, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    P_->fb = rp_begin(cb, &d);
    static VkClearRect tr[2048]; uint32_t nt = 0, ww = W.ext.width, hh = W.ext.height, cell = hh / 90 ? hh / 90 : 2;
    nt = text_rects(tr, nt, 2048, t1, (int32_t)((ww - strlen(t1) * 4 * cell) / 2), (int32_t)(hh / 2 - 9 * cell), cell);
    uint32_t c2 = cell * 2 / 3 ? cell * 2 / 3 : 1; nt = text_rects(tr, nt, 2048, t2, (int32_t)((ww - strlen(t2) * 4 * c2) / 2), (int32_t)(hh / 2 + 8 * cell), c2);
    uint32_t bw = ww * 6 / 10, bx = (ww - bw) / 2, by = hh / 2, bh = cell * 3, fw = total ? (uint32_t)((uint64_t)bw * done / total) : 0;
    VkClearRect outline = {{{(int32_t)bx, (int32_t)by}, {bw, bh}}, 0, 1}, fill = {{{(int32_t)bx, (int32_t)by}, {fw ? fw : 1, bh}}, 0, 1};
    VkClearAttachment white = {VK_IMAGE_ASPECT_COLOR_BIT, 0, {.color = {{0.85f, 0.85f, 0.9f, 1}}}}, dark = {VK_IMAGE_ASPECT_COLOR_BIT, 0, {.color = {{0.2f, 0.22f, 0.3f, 1}}}},
                      bar = {VK_IMAGE_ASPECT_COLOR_BIT, 0, {.color = {{0.35f, 0.7f, 1.0f, 1}}}};
    if (total) { ovl_clear(cb, &dark, 1, &outline); if (fw) ovl_clear(cb, &bar, 1, &fill); }
    if (nt) ovl_clear(cb, &white, nt, tr);
    rp_end(cb);
    img_barrier(cb, W.img[idx], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    vkEndCommandBuffer(cb);
    VkPipelineStageFlags ws = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 1, &P_->acq, &ws, 1, &cb, 1, &P_->done};
    VkPresentInfoKHR pi = {VK_STRUCTURE_TYPE_PRESENT_INFO_KHR, NULL, 1, &P_->done, 1, &W.sc, &idx, NULL};
    lk_(&g_qmu, LW_QMU); vkQueueSubmit(g_q, 1, &si, P_->fence); r = vkQueuePresentKHR(g_q, &pi); pthread_mutex_unlock(&g_qmu);
    if (r == VK_ERROR_OUT_OF_DATE_KHR) { W.gen = -1; W.why = "out of date at present"; } else if (r == VK_SUBOPTIMAL_KHR && W.gen != -1) win_suboptimal();
    front_window_lock(0);
}
int dm_vk_prewarm(void) {   // the front end, before the game's main: 0 when there is nothing to make
    const char *d = getenv("DM_VK_DIR"); if (!d) return 0;
    const char *e = getenv("DM_VK_PREWARM"); if (e && !atoi(e)) return 0;
    char p[1100]; snprintf(p, sizeof p, "%s/../pipeline_keys.bin", d); FILE *f = fopen(p, "rb"); if (!f) return 0;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint32_t n = sz > 0 ? (uint32_t)(sz / (long)sizeof(pso_rec)) : 0;
    if (!n || !(g_pw_recs = malloc((size_t)n * sizeof *g_pw_recs))) { fclose(f); return 0; }
    n = (uint32_t)fread(g_pw_recs, sizeof *g_pw_recs, n, f); fclose(f); if (!n) return 0;
    snprintf(p, sizeof p, "%s/../vk_depth_mode", d); if ((f = fopen(p, "r"))) { int m; if (fscanf(f, "%d", &m) == 1) { g_prewarm_depth = m != 0; g_depth_zero = g_prewarm_depth; } fclose(f); }
    vk_init(); g_prewarm_ran = 1; g_pw_total = n;
    long rss0 = 0; char ln[128]; FILE *st = fopen("/proc/self/status", "re"); if (st) { while (fgets(ln, sizeof ln, st)) sscanf(ln, "VmRSS: %ld", &rss0); fclose(st); }
    uint64_t t0 = now_ns(); int nw = 4; for (int i = 0; i < nw; i++) { pthread_t t; if (!pthread_create(&t, NULL, pw_worker, NULL)) pthread_detach(t); }
    LOG("vk: shader pre-warm: %u pipelines recorded by earlier runs, %d worker threads%s", n, nw, g_prewarm_depth >= 0 ? (g_prewarm_depth ? ", depth mode near = 0" : ", depth mode GL") : "");
    while (atomic_load(&g_pw_done) < n && !atomic_load(&g_pw_cancel)) { pw_draw(atomic_load(&g_pw_done), n); usleep(50000); }
    long rss1 = 0; if ((st = fopen("/proc/self/status", "re"))) { while (fgets(ln, sizeof ln, st)) sscanf(ln, "VmRSS: %ld", &rss1); fclose(st); }
    LOG("vk: shader pre-warm %s after %.1f s: %u of %u done, %llu pipelines made (%llu compiled, the rest from the driver's cache), %llu already there; %.0f ms loading shaders; RSS %+.0f MB%s",
        atomic_load(&g_pw_cancel) ? "skipped by a tap" : "finished", (now_ns() - t0) / 1e9, atomic_load(&g_pw_done), n, (unsigned long long)atomic_load(&g_pre_made),
        (unsigned long long)atomic_load(&g_pre_slow), (unsigned long long)atomic_load(&g_pre_had), atomic_load(&g_sh_ns) / 1e6, (rss1 - rss0) / 1024.0,
        atomic_load(&g_pw_cancel) ? "; the rest is made in the background" : "");
    return 1;
}
// ADPF (Android's performance hints): a hint session for the thread that records and presents (the game's render thread, which calls this), told each frame's work against a
// 16.7 ms target, and on Android 15 and later the GPU's part too, so the governors raise the CPU and GPU clocks for a game that keeps missing its target. GameTurbo showed the GPU
// 12% busy at 30 fps: the GPU clock stayed low, and our GPU timestamps grew with it. Looked up at run time (API 33 / 35 functions)
#include <dlfcn.h>
#include <dirent.h>
#include <stdbool.h>
#include <sys/syscall.h>
static uint64_t g_adpf_n, g_adpf_fail, g_adpf_cpu, g_adpf_gpu; static int g_adpf_err, g_adpf_v2;   // reports made, failed (and the last error), their CPU / GPU sums (the 600-frame line)
// Clocks and heat (vk19.45), every 600 frames: Run G ran at half the GPU and CPU time of the same build and scene another time, most likely with the device's performance mode
// on, so a run's numbers are only comparable with its clocks. The CPU frequencies (cpufreq, readable from the app), the GPU's where a readable source exists (tried in turn,
// the one found named once), the thermal status (AThermal, API 30) and headroom (API 31), the Android game mode the front end read (dm_game_mode: 0 unsupported, 1 standard,
// 2 performance, 3 battery, 4 custom) and the render thread's core
int dm_game_mode = -1;
static long rd_long(const char *path) { FILE *f = fopen(path, "re"); long v = -1; if (f) { if (fscanf(f, "%ld", &v) != 1) v = -1; fclose(f); } return v; }
// DM_TOPO_AFFINITY=1 (vk19.53, opt-in): host threads placed by the cores' measured strength, read at run time: cpu_capacity per core (else cpuinfo_max_freq; else nothing is
// done), grouped into tiers. The thread that records the submissions (the one in mtl_sub_begin: the game's RHI thread, or dm-record) goes to the strongest tier, the other hot
// threads (RHIThread, RenderThread, GameThread, RHIInterrupt by their names, and dm-complete) to the next one, everything else stays where the kernel puts it. No core number is
// assumed: on the Xiaomi Pad 8 the tiers came out as the device has them. A strongest tier of one core that is throttling (thermal status 2 or more, or headroom 0.95 or more)
// is widened to the top two tiers until it recovers. Host affinity only: the guest's core masks (core_serial) are untouched. Applied once a second to new or moved threads
_Atomic int g_rec_tid;
static int topo_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_TOPO_AFFINITY"); v = e && atoi(e) != 0; } return v; }
static cpu_set_t g_topo_top, g_topo_next, g_topo_top2; static int g_topo_ok = -1, g_topo_wide, g_topo_ntop; static long g_cap_top, g_cap_next;   // the two strongest capacities
static void topo_init(void) {
    long cap[64]; int n = (int)sysconf(_SC_NPROCESSORS_CONF); if (n > 64) n = 64; if (n < 1) n = 1;
    int src = 0;
    for (int c = 0; c < n; c++) { char pth[96]; snprintf(pth, sizeof pth, "/sys/devices/system/cpu/cpu%d/cpu_capacity", c); cap[c] = rd_long(pth); if (cap[c] > 0) src |= 1; }
    if (!(src & 1)) for (int c = 0; c < n; c++) { char pth[96]; snprintf(pth, sizeof pth, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", c); cap[c] = rd_long(pth); if (cap[c] > 0) src |= 2; }
    if (!src) { g_topo_ok = 0; LOG("vk: topology: neither cpu_capacity nor cpuinfo_max_freq readable: threads left where they are (DM_TOPO_AFFINITY)"); return; }
    long t1 = 0, t2 = 0; for (int c = 0; c < n; c++) if (cap[c] > t1) t1 = cap[c];
    for (int c = 0; c < n; c++) if (cap[c] < t1 && cap[c] > t2) t2 = cap[c];
    CPU_ZERO(&g_topo_top); CPU_ZERO(&g_topo_next); CPU_ZERO(&g_topo_top2); g_cap_top = t1; g_cap_next = t2;
    char b[256]; int o = 0;
    for (int c = 0; c < n; c++) { if (cap[c] == t1) { CPU_SET(c, &g_topo_top); CPU_SET(c, &g_topo_top2); g_topo_ntop++; } else if (cap[c] == t2) { CPU_SET(c, &g_topo_next); CPU_SET(c, &g_topo_top2); }
                                  o += snprintf(b + o, sizeof b - o, "%s%d:%ld", c ? " " : "", c, cap[c]); }
    if (!t2) g_topo_next = g_topo_top;   // one tier only: everything shares it
    g_topo_ok = 1; LOG("vk: topology (%s): %s; the recording thread on the %d core%s of the strongest tier, the other hot threads on the next", src & 1 ? "cpu_capacity" : "cpuinfo_max_freq", b, g_topo_ntop, g_topo_ntop > 1 ? "s" : "");
}
static struct { int tid; int role; } g_topo_set[32]; static int g_topo_nset;   // threads placed and how (role 1 recording, 2 hot)
// Guest threads by their guest name (hle_thread.c, at their start): Run Q matched names in /proc and took the app's own HWUI RenderThread, host threads that inherited a guest
// thread's name ("1002 GameThread" six times), and missed Unreal's render thread (its name is cut to "1019 RenderThre")
static struct { int tid; char name[32]; } g_gthr[128]; static int g_ngthr; static pthread_mutex_t g_gthr_mu = PTHREAD_MUTEX_INITIALIZER;
void dm_guest_thread_named(int tid, const char *name) {
    pthread_mutex_lock(&g_gthr_mu);
    int i = 0; while (i < g_ngthr && g_gthr[i].tid != tid) i++;
    if (i == g_ngthr && g_ngthr < 128) g_ngthr++;
    if (i < 128) { g_gthr[i].tid = tid; snprintf(g_gthr[i].name, sizeof g_gthr[i].name, "%s", name); }
    pthread_mutex_unlock(&g_gthr_mu);
}
static int topo_role_of(int tid) { for (int i = 0; i < g_topo_nset; i++) if (g_topo_set[i].tid == tid) return g_topo_set[i].role; return 0; }
static void mask_str(const cpu_set_t *m, char *b, size_t n) { int o = 0; b[0] = 0; for (int c = 0; c < 64 && o < (int)n - 4; c++) if (CPU_ISSET(c, m)) o += snprintf(b + o, n - o, "%s%d", o ? "," : "", c); }
static uint64_t g_topo_reapplied;
static void topo_place(int tid, int role, const char *name) {   // its mask, applied again whenever it is not what it should be (the system may move a thread's cpuset, which resets its mask)
    const cpu_set_t *m = role == 1 ? (g_topo_wide ? &g_topo_top2 : &g_topo_top) : &g_topo_next;
    cpu_set_t cur; CPU_ZERO(&cur); int known = !sched_getaffinity(tid, sizeof cur, &cur);
    int i = 0; while (i < g_topo_nset && g_topo_set[i].tid != tid) i++;
    int was = i < g_topo_nset ? g_topo_set[i].role : 0;
    if (known && was == role && CPU_EQUAL(&cur, m)) return;
    int r = sched_setaffinity(tid, sizeof *m, m), err = r ? errno : 0;
    if (i == g_topo_nset && g_topo_nset < 32) g_topo_nset++;
    if (i < 32) g_topo_set[i] = (typeof(g_topo_set[0])){tid, role};
    if (was == role) g_topo_reapplied++;
    cpu_set_t now; CPU_ZERO(&now); sched_getaffinity(tid, sizeof now, &now); char want[64], got[64]; mask_str(m, want, sizeof want); mask_str(&now, got, sizeof got);
    static int nl; if (nl++ < 40 || r) {
        char cs[96] = ""; if (r) { char pth[64]; snprintf(pth, sizeof pth, "/proc/self/task/%d/cpuset", tid); FILE *f = fopen(pth, "re"); if (f) { if (fgets(cs, sizeof cs, f)) cs[strcspn(cs, "\n")] = 0; fclose(f); } }
        LOG("vk: topology: %s (tid %d) %s: wanted {%s}, has {%s}%s%s%s%s", name, tid, role == 1 ? "the recording thread" : "a hot thread", want, got, was == role ? " (applied again: it had been changed)" : "",
            r ? " -- sched_setaffinity failed: " : "", r ? strerror(err) : "", r && cs[0] ? (snprintf(cs + strlen(cs), sizeof cs - strlen(cs), " (cpuset)"), cs) : "");
    }
}
static void topo_tick(void) {
    if (!topo_on()) return;
    if (g_topo_ok < 0) topo_init();
    if (!g_topo_ok) return;
    {   // a single strongest core throttling: the top two tiers until it recovers
        static void *tm; static int (*st)(void *); static float (*hr)(void *, int); static int ti;
        if (!ti) { ti = 1; void *h = dlopen("libandroid.so", RTLD_NOW); void *(*acq)(void) = h ? (void *(*)(void))dlsym(h, "AThermal_acquireManager") : NULL;
                   st = h ? (int (*)(void *))dlsym(h, "AThermal_getCurrentThermalStatus") : NULL; hr = h ? (float (*)(void *, int))dlsym(h, "AThermal_getThermalHeadroom") : NULL; tm = acq ? acq() : NULL; }
        int ts = tm && st ? st(tm) : 0; float hd = tm && hr ? hr(tm, 0) : 0;
        int wide = g_topo_wide;
        if (g_topo_ntop == 1 && (ts >= 2 || hd >= 0.95f)) wide = 1; else if (ts < 2 && hd < 0.85f) wide = 0;
        if (wide != g_topo_wide) { g_topo_wide = wide; LOG("vk: topology: thermal status %d, headroom %.2f: the recording thread %s", ts, (double)hd, wide ? "widened to the top two tiers" : "back on the strongest tier");
                                   for (int i = 0; i < g_topo_nset; i++) if (g_topo_set[i].role == 1) g_topo_set[i].role = -1; }   // re-placed below
    }
    int rec = atomic_load(&g_rec_tid);
    // the guest's hot threads by their guest names (dm_guest_thread_named), the backend's by their own (dm-complete, dm-record)
    static const char *const ghot[] = {"RHIThread", "RenderThread", "GameThread", "RHIInterrupt"};
    struct { int tid; char name[32]; } lst[160]; int nl = 0;
    pthread_mutex_lock(&g_gthr_mu);
    for (int i = 0; i < g_ngthr && nl < 160; i++) for (size_t k = 0; k < sizeof ghot / sizeof *ghot; k++) if (!strncmp(g_gthr[i].name, ghot[k], strlen(ghot[k]))) {
        lst[nl].tid = g_gthr[i].tid; snprintf(lst[nl].name, sizeof lst[nl].name, "%s", g_gthr[i].name); nl++; break; }
    pthread_mutex_unlock(&g_gthr_mu);
    DIR *d = opendir("/proc/self/task");
    if (d) { for (struct dirent *e; (e = readdir(d)) && nl < 160;) {
        int tid = atoi(e->d_name); if (tid <= 0) continue;
        char pth[64], cm[32] = ""; snprintf(pth, sizeof pth, "/proc/self/task/%d/comm", tid); FILE *f = fopen(pth, "re"); if (!f) continue;
        if (fgets(cm, sizeof cm, f)) cm[strcspn(cm, "\n")] = 0; fclose(f);
        if (!strcmp(cm, "dm-complete") || !strcmp(cm, "dm-record") || tid == rec) { lst[nl].tid = tid; snprintf(lst[nl].name, sizeof lst[nl].name, "%s", cm); nl++; }
      } closedir(d); }
    for (int i = 0; i < nl; i++) { int dup = 0; for (int j = 0; j < i; j++) dup |= lst[j].tid == lst[i].tid; if (dup) continue;
        char pth[64]; snprintf(pth, sizeof pth, "/proc/self/task/%d", lst[i].tid); if (access(pth, F_OK)) continue;   // gone
        topo_place(lst[i].tid, lst[i].tid == rec ? 1 : 2, lst[i].name); }
}
// DM_TOPO_SELF=1 (vk19.55, opt-in): the recording thread puts ITSELF on the strongest tier. Run R showed another thread's mask set by sched_setaffinity(tid) read back as every core
// right away, while the runs whose recording thread sat on the strongest core (Q, Q': cpu7 in 20 of 20 samples) recorded 4 ms faster and ran 3 fps faster than those where it
// sat on the gold cores (J', R, S). The recording thread is the one with the most submissions over the last 300 (and at least 30 of them in a row); from inside mtl_sub_begin it
// calls sched_setaffinity(0) with the strongest tier (the top two while a single strongest core throttles), reads its own mask back every 60 submissions and applies it again
// when it changed (counted). No other thread is touched
static int topo_self_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_TOPO_SELF"); v = e && atoi(e) != 0; if (v) LOG("vk: DM_TOPO_SELF=1: the recording thread places itself on the strongest cores"); } return v; }
static uint64_t g_ts_applied, g_ts_reapplied;
// DM_UCLAMP_MIN=<0..1024> or auto (vk19.56, opt-in): the recording thread (identified as above) sets its own utilisation floor with sched_setattr(SCHED_FLAG_UTIL_CLAMP_MIN, keeping
// its policy and nice). App affinity is a no-op on this kernel (U1, U2: even the thread's own sched_setaffinity reads back as every core), but the energy-aware scheduler places
// a task by its utilisation against each core's capacity and sets the frequency from it: a floor above the second-strongest tier's capacity makes the scheduler itself prefer the
// strongest cores and keep their clock up. auto: halfway between the two strongest capacities, read at run time. Read back with sched_getattr; logged with errno; checked again
// every 600 submissions. Also logged once, for the record: whether the cpu controller's uclamp files are writable
struct dm_sched_attr { uint32_t size, sched_policy; uint64_t sched_flags; int32_t sched_nice; uint32_t sched_priority; uint64_t sched_runtime, sched_deadline, sched_period; uint32_t sched_util_min, sched_util_max; };
static long uclamp_want(void) {   // -1: off
    static long v = -2; if (v != -2) return v;
    const char *e = getenv("DM_UCLAMP_MIN"); if (!e || !*e) return v = -1;
    if (!strcmp(e, "auto")) { if (g_topo_ok < 0) topo_init(); v = g_topo_ok && g_cap_top > 0 ? (g_cap_next > 0 ? (g_cap_top + g_cap_next) / 2 : g_cap_top * 3 / 4) : 768; }
    else { v = atol(e); if (v < 0) v = 0; if (v > 1024) v = 1024; }
    LOG("vk: DM_UCLAMP_MIN=%s: the recording thread's utilisation floor %ld (capacities: strongest %ld, next %ld)", e, v, g_cap_top, g_cap_next);
    return v;
}
static void uclamp_apply(int tid, int again) {
    long want = uclamp_want(); if (want < 0) return;
    struct dm_sched_attr a; memset(&a, 0, sizeof a); a.size = sizeof a;
    if (again && syscall(SYS_sched_getattr, 0, &a, sizeof a, 0) == 0 && a.sched_util_min == (uint32_t)want) return;
    memset(&a, 0, sizeof a); a.size = sizeof a; a.sched_flags = 0x18 /* KEEP_POLICY | KEEP_PARAMS */ | 0x20 /* UTIL_CLAMP_MIN */; a.sched_util_min = (uint32_t)want;
    long r = syscall(SYS_sched_setattr, 0, &a, 0); int err = r ? errno : 0;
    struct dm_sched_attr g; memset(&g, 0, sizeof g); g.size = sizeof g; long rg = syscall(SYS_sched_getattr, 0, &g, sizeof g, 0);
    static int nl; if (nl++ < 12 || r) LOG("vk: DM_UCLAMP_MIN: tid %d %s: sched_setattr %s%s; reads back util_min %u, util_max %u (sched_getattr %s)", tid, again ? "again" : "set", r ? "failed: " : "0", r ? strerror(err) : "",
                                        rg ? 0 : g.sched_util_min, rg ? 0 : g.sched_util_max, rg ? strerror(errno) : "0");
    static int once; if (!once++) {   // for the record: the cpu controller's clamps
        static const char *const f[] = {"/dev/cpuctl/top-app/cpu.uclamp.min", "/dev/cpuctl/cpu.uclamp.min", "/sys/fs/cgroup/cpu.uclamp.min"};
        char cg[160] = ""; FILE *fc = fopen("/proc/self/cgroup", "re"); if (fc) { char ln[160]; while (fgets(ln, sizeof ln, fc)) if (strstr(ln, "cpu")) { ln[strcspn(ln, "\n")] = 0; snprintf(cg, sizeof cg, "%s", ln); break; } fclose(fc); }
        LOG("vk: DM_UCLAMP_MIN: cgroup '%s'; %s %s, %s %s, %s %s", cg, f[0], access(f[0], W_OK) ? "not writable" : "writable", f[1], access(f[1], W_OK) ? "not writable" : "writable", f[2], access(f[2], W_OK) ? "not writable" : "writable");
    }
}
void topo_self_tick(void) {
    if (!topo_self_on() && uclamp_want() < 0) return;
    static int win_tid[8], win_cnt[8], win_n, subs, cand, streak, best_streak[8], winner; static uint64_t nsub;
    int t = (int)gettid(); nsub++;
    if (t != cand) { cand = t; streak = 0; } streak++;
    int k = 0; while (k < win_n && win_tid[k] != t) k++;
    if (k == win_n && win_n < 8) { win_tid[k] = t; win_cnt[k] = 0; best_streak[k] = 0; win_n++; }
    if (k < 8) { win_cnt[k]++; if (streak > best_streak[k]) best_streak[k] = streak; }
    if (++subs >= 300) {   // the window's winner: most submissions, and a run of 30 at least
        int w = -1; for (int i = 0; i < win_n; i++) if (best_streak[i] >= 30 && (w < 0 || win_cnt[i] > win_cnt[w])) w = i;
        if (w >= 0 && win_tid[w] != winner) { LOG("vk: DM_TOPO_SELF: the recording thread is tid %d (%d of the last 300 submissions)", win_tid[w], win_cnt[w]); winner = win_tid[w]; }
        subs = 0; win_n = 0;
    }
    if (t != winner) return;
    if (uclamp_want() >= 0) { static uint64_t last_uc; static int uc_tid; if (uc_tid != t || nsub - last_uc >= 600) { uclamp_apply(t, uc_tid == t); uc_tid = t; last_uc = nsub; } }
    if (!topo_self_on()) return;
    if (g_topo_ok < 0) topo_init();
    if (!g_topo_ok) return;
    static uint64_t last_chk, last_th; static int placed, wide;
    if (nsub - last_th >= 600) { last_th = nsub;   // a single strongest core throttling: the top two tiers until it recovers
        static void *tm; static int (*st)(void *); static float (*hr)(void *, int); static int ti;
        if (!ti) { ti = 1; void *h = dlopen("libandroid.so", RTLD_NOW); void *(*acq)(void) = h ? (void *(*)(void))dlsym(h, "AThermal_acquireManager") : NULL;
                   st = h ? (int (*)(void *))dlsym(h, "AThermal_getCurrentThermalStatus") : NULL; hr = h ? (float (*)(void *, int))dlsym(h, "AThermal_getThermalHeadroom") : NULL; tm = acq ? acq() : NULL; }
        int ts = tm && st ? st(tm) : 0; float hd = tm && hr ? hr(tm, 0) : 0; int w2 = wide;
        if (g_topo_ntop == 1 && (ts >= 2 || hd >= 0.95f)) w2 = 1; else if (ts < 2 && hd < 0.85f) w2 = 0;
        if (w2 != wide) { wide = w2; placed = 0; LOG("vk: DM_TOPO_SELF: thermal status %d, headroom %.2f: %s", ts, (double)hd, wide ? "on the top two tiers" : "back on the strongest tier"); }
    }
    if (placed && nsub - last_chk < 60) return;
    last_chk = nsub;
    const cpu_set_t *m = wide ? &g_topo_top2 : &g_topo_top;
    cpu_set_t cur; CPU_ZERO(&cur);
    if (placed && !sched_getaffinity(0, sizeof cur, &cur) && CPU_EQUAL(&cur, m)) return;
    int r = sched_setaffinity(0, sizeof *m, m), err = r ? errno : 0;
    cpu_set_t now; CPU_ZERO(&now); sched_getaffinity(0, sizeof now, &now); char want[64], got[64]; mask_str(m, want, sizeof want); mask_str(&now, got, sizeof got);
    if (placed) g_ts_reapplied++; else g_ts_applied++;
    static int nl; if (nl++ < 20 || r) LOG("vk: DM_TOPO_SELF: tid %d %s: wanted {%s}, reads back {%s}%s%s", t, placed ? "placed again (its mask had changed)" : "placed", want, got, r ? " -- failed: " : "", r ? strerror(err) : "");
    placed = 1;
}
static void clock_line(void) {
    char b[256]; int o = 0;
    for (int c = 0; c < 8; c++) { char pth[96]; snprintf(pth, sizeof pth, "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", c); long f = rd_long(pth); o += snprintf(b + o, sizeof b - o, "%s%ld", c ? " " : "", f > 0 ? f / 1000 : -1); }
    static const char *const gp[] = {"/sys/class/kgsl/kgsl-3d0/devfreq/cur_freq", "/sys/class/kgsl/kgsl-3d0/gpuclk", "/sys/kernel/gpu/gpu_clock", "/sys/class/devfreq/gpufreq/cur_freq", NULL};
    static int gi = -2; long g = -1;
    if (gi == -2) { gi = -1; for (int i = 0; gp[i]; i++) if (rd_long(gp[i]) > 0) { gi = i; LOG("vk: the GPU clock is read from %s", gp[i]); break; } if (gi < 0) LOG("vk: no readable GPU clock (kgsl sysfs and /sys/kernel/gpu denied or absent)"); }
    if (gi >= 0) { g = rd_long(gp[gi]); if (g > 100000000) g /= 1000000; else if (g > 100000) g /= 1000; }   // Hz or kHz to MHz
    static void *tm; static int (*st)(void *); static float (*hr)(void *, int); static int ti;
    if (!ti) { ti = 1; void *h = dlopen("libandroid.so", RTLD_NOW);
               void *(*acq)(void) = h ? (void *(*)(void))dlsym(h, "AThermal_acquireManager") : NULL; st = h ? (int (*)(void *))dlsym(h, "AThermal_getCurrentThermalStatus") : NULL;
               hr = h ? (float (*)(void *, int))dlsym(h, "AThermal_getThermalHeadroom") : NULL; tm = acq ? acq() : NULL; }
    int ts = tm && st ? st(tm) : -1; float hd = tm && hr ? hr(tm, 0) : -1.0f;
    char hb[256] = ""; int ho = 0;   // where the hot threads ran last (DM_TOPO_AFFINITY's placement, or the kernel's)
    for (int i = 0; i < g_topo_nset && ho < (int)sizeof hb - 40; i++) { char pth[64], ln[512]; snprintf(pth, sizeof pth, "/proc/self/task/%d/stat", g_topo_set[i].tid); FILE *f = fopen(pth, "re");
        if (f && fgets(ln, sizeof ln, f)) { char *q = strrchr(ln, ')'); int fld = 0, cpu = -1; for (char *t = q ? q + 2 : ln; *t && fld < 37; t++) if (*t == ' ') { fld++; if (fld == 36) cpu = atoi(t + 1); }
                                            ho += snprintf(hb + ho, sizeof hb - ho, "%s%d:%d", ho ? " " : "; placed threads tid:core ", g_topo_set[i].tid, cpu); }
        if (i == g_topo_nset - 1 && g_topo_reapplied && ho < (int)sizeof hb - 40) ho += snprintf(hb + ho, sizeof hb - ho, " (masks applied again %llu times)", (unsigned long long)g_topo_reapplied);
        if (f) fclose(f); }
    if (topo_self_on() && (int)sizeof hb - (int)strlen(hb) > 48) snprintf(hb + strlen(hb), sizeof hb - strlen(hb), "; self-placed %llu, again %llu", (unsigned long long)g_ts_applied, (unsigned long long)g_ts_reapplied);
    LOG("vk: clocks: CPU MHz by core %s; GPU %ld MHz; thermal status %d, headroom %.2f; game mode %d; this thread on core %d (tid %d)%s", b, g, ts, (double)hd, dm_game_mode, sched_getcpu(), (int)gettid(), hb);
}
// DM_ADPF_ALL=1 (vk19.50, opt-in): the hint session holds every hot thread, not only the one that presents: the game's RHIThread, RenderThread, GameThread and RHIInterrupt, our
// dm-complete and dm-record (found by name in /proc/self/task, again every 600 frames: APerformanceHint_setThreads, API 34), asks not to be run for power efficiency
// (APerformanceHint_setPreferPowerEfficiency(false), API 35), and on a frame over 1.5 times the target says the work grew (APerformanceHint_notifyWorkloadIncrease, API 36), at most
// once a second. The clocks sagged to 787 MHz on the gold cores and 1.67 GHz on the prime one while the session reported 43 ms against 16.7. No core is named anywhere
static int adpf_all(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_ADPF_ALL"); v = e && atoi(e) != 0; } return v; }
static int adpf_tids(int32_t *out, int max, char *names, size_t nn) {   // the hot threads by name, this one first
    static const char *const want[] = {"RHIThread", "RenderThread", "GameThread", "RHIInterrupt", "dm-complete", "dm-record"};
    int n = 0; out[n++] = (int32_t)syscall(SYS_gettid); size_t o = 0; if (nn) names[0] = 0;
    DIR *d = opendir("/proc/self/task"); if (!d) return n;
    for (struct dirent *e; (e = readdir(d)) && n < max;) {
        int tid = atoi(e->d_name); if (tid <= 0 || tid == out[0]) continue;
        char pth[64], cm[32] = ""; snprintf(pth, sizeof pth, "/proc/self/task/%d/comm", tid); FILE *f = fopen(pth, "re"); if (!f) continue;
        if (fgets(cm, sizeof cm, f)) cm[strcspn(cm, "\n")] = 0; fclose(f);
        for (size_t k = 0; k < sizeof want / sizeof *want; k++) if (strstr(cm, want[k])) { out[n++] = tid; if (o < nn) o += (size_t)snprintf(names + o, nn - o, "%s%s", o ? ", " : "", cm); break; }
    }
    closedir(d); return n;
}
static void adpf_frame(uint64_t cpu_ns, uint64_t gpu_ns) {
    static int (*set_threads)(void *, const int32_t *, size_t); static int (*notify_up)(void *, bool, bool, const char *); static uint64_t last_up, nframe;
    static int state; static void *session; static void *(*mk_work)(void); static void (*w_start)(void *, int64_t), (*w_total)(void *, int64_t), (*w_cpu)(void *, int64_t), (*w_gpu)(void *, int64_t), (*w_free)(void *);
    static int (*report)(void *, int64_t); static int (*report2)(void *, void *);
    if (state < 0) return;
    if (!state) {
        state = -1; const char *e = getenv("DM_ADPF"); if (e && !atoi(e)) { LOG("vk: ADPF off (DM_ADPF=0)"); return; }
        void *h = dlopen("libandroid.so", RTLD_NOW); if (!h) { LOG("[RED] vk: ADPF: libandroid.so not opened"); return; }
        void *(*getm)(void) = (void *(*)(void))dlsym(h, "APerformanceHint_getManager");
        void *(*mks)(void *, const int32_t *, size_t, int64_t) = (void *(*)(void *, const int32_t *, size_t, int64_t))dlsym(h, "APerformanceHint_createSession");
        report = (int (*)(void *, int64_t))dlsym(h, "APerformanceHint_reportActualWorkDuration");
        report2 = (int (*)(void *, void *))dlsym(h, "APerformanceHint_reportActualWorkDuration2");
        mk_work = (void *(*)(void))dlsym(h, "AWorkDuration_create"); w_free = (void (*)(void *))dlsym(h, "AWorkDuration_release");
        w_start = (void (*)(void *, int64_t))dlsym(h, "AWorkDuration_setWorkPeriodStartTimestampNanos"); w_total = (void (*)(void *, int64_t))dlsym(h, "AWorkDuration_setActualTotalDurationNanos");
        w_cpu = (void (*)(void *, int64_t))dlsym(h, "AWorkDuration_setActualCpuDurationNanos"); w_gpu = (void (*)(void *, int64_t))dlsym(h, "AWorkDuration_setActualGpuDurationNanos");
        if (!getm || !mks || !report) { LOG("[RED] vk: ADPF: the performance hint functions are missing (API 33+)"); return; }
        void *m = getm(); int32_t tids[16]; char nm[256] = ""; int nt = adpf_all() ? adpf_tids(tids, 16, nm, sizeof nm) : (tids[0] = (int32_t)syscall(SYS_gettid), 1);
        if (!m || !(session = mks(m, tids, (size_t)nt, 16666667))) { LOG("[RED] vk: ADPF: no hint session (manager %p): the device gives none, the clocks follow their own governor", m); return; }
        g_adpf_v2 = report2 && mk_work && w_gpu;
        state = 1; LOG("vk: ADPF hint session for %d thread%s (tid %d%s%s), target 16.7 ms%s", nt, nt > 1 ? "s" : "", tids[0], nt > 1 ? " and " : "", nm, report2 && mk_work && w_gpu ? ", with the GPU's work (Android 15)" : "");
        if (adpf_all()) {
            set_threads = (int (*)(void *, const int32_t *, size_t))dlsym(h, "APerformanceHint_setThreads");
            int (*pref)(void *, bool) = (int (*)(void *, bool))dlsym(h, "APerformanceHint_setPreferPowerEfficiency");
            notify_up = (int (*)(void *, bool, bool, const char *))dlsym(h, "APerformanceHint_notifyWorkloadIncrease");
            int rp = pref ? pref(session, false) : -1;
            LOG("vk: ADPF (DM_ADPF_ALL): setThreads %s, setPreferPowerEfficiency(false) %s (%d), notifyWorkloadIncrease %s", set_threads ? "available" : "missing (API 34)", pref ? "called" : "missing (API 35)", rp, notify_up ? "available" : "missing (API 36)");
        }
    }
    nframe++;
    if (adpf_all() && set_threads && nframe % 600 == 0) {   // threads come and go (the record thread, a new RHI thread after a reset): the list again
        int32_t tids[16]; char nm[256] = ""; int nt = adpf_tids(tids, 16, nm, sizeof nm); int r = set_threads(session, tids, (size_t)nt);
        static int nl; if (nl++ < 6 || r) LOG("vk: ADPF threads now %d (%s): setThreads %d", nt, nm, r);
    }
    if (adpf_all() && notify_up && cpu_ns > 25000000ull) { uint64_t t = now_ns(); if (t - last_up > 1000000000ull) { last_up = t; int r = notify_up(session, true, gpu_ns > 25000000ull, "frame over target"); static int nl; if (nl++ < 3) LOG("vk: ADPF notifyWorkloadIncrease %d", r); } }
    if (report2 && mk_work && w_start && w_total && w_cpu && w_gpu) {
        void *w = mk_work(); if (!w) return;
        uint64_t now = now_ns(), tot = cpu_ns > gpu_ns ? cpu_ns : gpu_ns;
        w_start(w, (int64_t)(now - tot)); w_total(w, (int64_t)tot); w_cpu(w, (int64_t)cpu_ns); w_gpu(w, (int64_t)gpu_ns); int r = report2(session, w); if (w_free) w_free(w);
        g_adpf_n++; g_adpf_cpu += cpu_ns; g_adpf_gpu += gpu_ns; if (r) { g_adpf_fail++; g_adpf_err = r; }
    } else { int r = report(session, (int64_t)cpu_ns); g_adpf_n++; g_adpf_cpu += cpu_ns; if (r) { g_adpf_fail++; g_adpf_err = r; } }
}
// The vk19.59 counters (improvement plan v4, step 0), every 600 frames: copies, index ranges, texel views and tex_get, the shadow mode, present-to-present intervals, the
// recording thread's cores, page faults by thread and the guest-memory protection's mprotect time
static uint32_t g_pp[600]; static uint64_t g_npp, g_pp_last;
static int u32cmp(const void *a, const void *b) { uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b; return x < y ? -1 : x > y; }
// H0 sawtooth: once a frame (the presenting thread). Every 100 frames a line with the frame's averages over those 100 (at most 2,000 lines); a line when the chained dispatch
// bindings stop or start, or the live GPU-written bytes fall by half (at most 40 each)
static void h0_tick(uint64_t n) {
    if (!h0_on()) return;
    static uint64_t f0, chain_prev = UINT64_MAX, gwb_prev, pf_ch, pf_gwb;   // per frame: the previous frame's values, the counters at the previous frame
    static uint64_t b_ch, b_chb, b_gwn, b_gwb, b_cpd, b_cpp, b_cpv, b_cmb; static int nl, nlc, nlg;   // per 100 frames: the counters at the last line
    uint64_t dch = g_h0_chain_n - pf_ch, dgwb = g_h0_gw_b - pf_gwb; pf_ch = g_h0_chain_n; pf_gwb = g_h0_gw_b;
    if (chain_prev != UINT64_MAX && ((chain_prev && !dch) || (!chain_prev && dch)) && nlc++ < 40)
        LOG("vk: H0 frame %llu: dispatches bound from a running submission's GPU-written ranges %s (%llu this frame, %llu the frame before)", (unsigned long long)n, dch ? "again" : "no longer", (unsigned long long)dch, (unsigned long long)chain_prev);
    if (chain_prev != UINT64_MAX && gwb_prev > (1u << 20) && dgwb * 2 < gwb_prev && nlg++ < 40)
        LOG("vk: H0 frame %llu: live GPU-written bytes at submission start fell to %.2f MB (%.2f MB the frame before)", (unsigned long long)n, dgwb / 1048576.0, gwb_prev / 1048576.0);
    chain_prev = dch; gwb_prev = dgwb;
    if (!f0) { f0 = n; b_ch = g_h0_chain_n; b_chb = g_h0_chain_b; b_gwn = g_h0_gw_n; b_gwb = g_h0_gw_b; b_cpd = g_h0_cp_draws; b_cpp = g_h0_cp_prims; b_cpv = g_h0_cp_vbytes; b_cmb = g_h0_cmp_bytes; return; }
    if (n - f0 < 100) return;
    double k = (double)(n - f0); f0 = n;
    uint64_t id = atomic_exchange(&g_h0_ii_draws, 0), ii = atomic_exchange(&g_h0_ii_inst, 0), ip = atomic_exchange(&g_h0_ii_prims, 0), im = atomic_exchange(&g_h0_ii_max, 0), icp = atomic_exchange(&g_h0_ii_cp_inst, 0);
    uint64_t lb = atomic_exchange(&g_h0_lost_b, 0), ln = atomic_exchange(&g_h0_lost_n, 0), sk = atomic_exchange(&g_h0_snap_skip, 0);
    uint64_t ifi = atomic_exchange(&g_h0_ii_fi, 0), ibv = atomic_exchange(&g_h0_ii_bv, 0), ppi = atomic_exchange(&g_h0_past_i, 0), ppv = atomic_exchange(&g_h0_past_v, 0);
    if (nl < 2000) LOG("vk: H0 frame %llu, base instances per frame over the last %.0f: GPU-written indirect draws with a first instance not 0 %.1f, with a vertex offset not 0 %.1f; other draws with a base instance not 0 %.1f; GPU-written indirect draws reading past the index window %.2f, past the vertex copy %.2f",
                       (unsigned long long)n, k, ifi / k, ibv / k, g_h0_dbi / k, ppi / k, ppv / k);
    g_h0_dbi = 0;
    if (nl++ < 2000)
        LOG("vk: H0 frame %llu, per frame over the last %.0f: GPU-written entries at submission start %.1f (%.2f MB; most %llu, %.2f MB); dispatch storage buffers bound from a running submission's range %.1f (%.2f MB), from guest memory %.1f; their whole write-backs: CPU writes overwritten %.0f bytes in %.1f ranges (%llu too big to check); composes %.2f MB (this submission's parts %.2f MB); composed passes' draws %.1f (direct vertices x instances %.0f, bound vertex bytes %.1f MB); GPU-written indirect draws %.1f, instances %.0f (in composed passes %.0f), vertices x instances %.0f, most instances in one %llu",
            (unsigned long long)n, k, (g_h0_gw_n - b_gwn) / k, (g_h0_gw_b - b_gwb) / k / 1048576.0, (unsigned long long)g_h0_gw_nmax, g_h0_gw_bmax / 1048576.0,
            (g_h0_chain_n - b_ch) / k, (g_h0_chain_b - b_chb) / k / 1048576.0, g_h0_dsp_fresh / k, lb / k, ln / k, (unsigned long long)sk,
            (g_h0_cmp_bytes - b_cmb) / k / 1048576.0, g_h0_cmp_cur_bytes / k / 1048576.0, (g_h0_cp_draws - b_cpd) / k, (g_h0_cp_prims - b_cpp) / k, (g_h0_cp_vbytes - b_cpv) / k / 1048576.0,
            id / k, ii / k, icp / k, ip / k, (unsigned long long)im);
    b_ch = g_h0_chain_n; b_chb = g_h0_chain_b; b_gwn = g_h0_gw_n; b_gwb = g_h0_gw_b; b_cpd = g_h0_cp_draws; b_cpp = g_h0_cp_prims; b_cpv = g_h0_cp_vbytes; b_cmb = g_h0_cmp_bytes;
    g_h0_dsp_fresh = 0; g_h0_cmp_cur_bytes = 0; g_h0_gw_nmax = g_h0_gw_bmax = 0;
}
// H0 (Harvest Moon plan v2): the per-600 lines; counters zeroed after. Called on the presenting thread with the other per-600 lines
static void h0_lines(void) {
    if (gw_shadow_on()) { LOG("vk: DM_VK_GW_SHADOW per frame over the last 600: aligned copies made after dispatches %.1f, reads remembered (new or changed) %.2f", g_shl_made / 600.0, g_shl_learn / 600.0); g_shl_made = g_shl_learn = 0; }
    if (gw_pad_on()) { LOG("vk: DM_VK_GW_PAD per frame over the last 600: windows made %.1f, small copies into a window %.1f, earlier submissions' parts put into new windows %.1f; not padded: no pool %.1f, this submission's other GPU writes there %.1f, no room %.1f",
                           g_pad_n / 600.0, g_pad_into / 600.0, g_pad_over / 600.0, g_pad_why[1] / 600.0, g_pad_why[2] / 600.0, g_pad_why[3] / 600.0); g_pad_n = g_pad_into = g_pad_over = 0; memset(g_pad_why, 0, sizeof g_pad_why); }
    if (idxpool_once_mode()) { LOG("vk: DM_VK_IDXPOOL_ONCE per frame over the last 600: pool copies %.2f (%.2f MB), draws bound to one %.1f, left to the old copy %.2f, copied again after a write-back landed in it %.2f, verify differences %.2f",
                                   g_ipo_copy / 600.0, g_ipo_bytes / 600.0 / 1048576.0, g_ipo_hit / 600.0, g_ipo_fallback / 600.0, g_ipo_hwl / 600.0, g_ipo_bad / 600.0); g_ipo_copy = g_ipo_bytes = g_ipo_hit = g_ipo_fallback = g_ipo_bad = g_ipo_hwl = 0; }
    if (ring_reuse_on()) { LOG("vk: DM_VK_RING_REUSE over the last 600 frames: rings of idle contexts given back %llu, extra rings sized from the overflow %llu (recent overflow %.1f MB); %d extra rings kept (%.0f MB)",
                               (unsigned long long)g_h05_shrunk, (unsigned long long)g_h05_sized, g_h05_over / 1048576.0, g_nxc, g_xc_bytes / 1048576.0); g_h05_shrunk = g_h05_sized = 0; }
    if (chain_snap_mode()) { uint64_t pb = atomic_exchange(&g_cs_prop_b, 0), pn = atomic_exchange(&g_cs_prop_n, 0), kw = atomic_exchange(&g_cs_kept_w, 0), bw = atomic_exchange(&g_cs_both_w, 0), wn = atomic_exchange(&g_cs_wb_n, 0);
        LOG("vk: chained GPU-written ranges per frame over the last 600 (DM_VK_GW_CHAIN_SNAP %d): given a snapshot %.1f (%.2f MB), written back by changed words %.1f; earlier write-backs put into later snapshots %.0f bytes (%.1f snapshots); buffer copies of GPU-written bytes kept out of the CPU's rewrites %.1f",
            chain_snap_mode(), g_cs_n / 600.0, g_cs_b / 600.0 / 1048576.0, wn / 600.0, pb / 600.0, pn / 600.0, g_cs_b2b / 600.0); g_cs_b2b = 0;
        if (chain_snap_mode() == 2) LOG("vk: chained GPU-written ranges (verify): the CPU's words kept that the whole write-back overwrote before %.0f bytes a frame; words both the CPU and the GPU wrote (the GPU's kept) %.0f bytes a frame", kw * 4 / 600.0, bw * 4 / 600.0);
        g_cs_n = g_cs_b = 0; }
    if (!h0_on()) return;   // the H0 lines only with DM_VK_H0=1
    static const char *cn[H0C_N] = {"vertex", "uniform", "storage", "texel", "indirect", "buffer copy", "image copy", "other"};
    char b0[512], b1[512]; int o0 = 0, o1 = 0;
    for (int i = 0; i < H0C_N; i++) { o0 += snprintf(b0 + o0, sizeof b0 - o0, " %s %.1f", cn[i], g_h0_cmp[0][i] / 600.0); o1 += snprintf(b1 + o1, sizeof b1 - o1, " %s %.1f", cn[i], g_h0_cmp[1][i] / 600.0); }
    LOG("vk: H0 composes per frame over the last 600: over this submission's GPU-written ranges by caller:%s; over earlier submissions' only:%s; passes ended for them %.1f",
        b1, b0, g_h0_cmp_pass / 600.0);
    LOG("vk: H0 compose reasons per frame (a read may count under several): past the entry's end by <=16 bytes %.1f, by <=256 %.1f, by more %.1f; before its start %.1f; inside but misaligned for the binding %.1f; over 2 or more entries %.1f; the entry is an exact copy's %.1f",
        g_h0_cmp_why[H0R_PAST16] / 600.0, g_h0_cmp_why[H0R_PAST256] / 600.0, g_h0_cmp_why[H0R_PASTMORE] / 600.0, g_h0_cmp_why[H0R_BEFORE] / 600.0, g_h0_cmp_why[H0R_MISALIGN] / 600.0,
        g_h0_cmp_why[H0R_MULTI] / 600.0, g_h0_cmp_why[H0R_EXACT] / 600.0);
    LOG("vk: H0 generations per frame: new submission %.2f, extra ring %.2f, write log full %.2f; copy_in 'earlier submission' misses: an earlier submission's copy %.1f, this submission's (ring change or log overflow) %.1f; write-log entries scanned per found key: copy_in %.1f, index ranges %.1f; index ranges scanned again: after a new submission %.1f, within one %.1f",
        g_h0_gen[0] / 600.0, g_h0_gen[1] / 600.0, g_h0_gen[2] / 600.0, g_h0_mrgen[0] / 600.0, g_h0_mrgen[1] / 600.0,
        g_h0_wscan_n[0] ? (double)g_h0_wscan[0] / g_h0_wscan_n[0] : 0.0, g_h0_wscan_n[1] ? (double)g_h0_wscan[1] / g_h0_wscan_n[1] : 0.0, g_h0_ic_rescan[0] / 600.0, g_h0_ic_rescan[1] / 600.0);
    { static const char *ccn[CI_N] = {"vertex", "index", "index pool", "uniform", "storage", "texel", "image copy"}; char b[1024]; int o = 0;
      for (int c = 0; c < CI_N && c < 8; c++) { o += snprintf(b + o, sizeof b - o, "%s %s", c ? ";" : "", ccn[c]);
          for (int w = 0; w < MR_N && w < 8; w++) o += snprintf(b + o, sizeof b - o, " %.1f", g_h0_ci[c][w] / 600.0); }
      LOG("vk: H0 copy_in misses per frame by caller (free slot / held by another range / generation / larger / alignment / written since):%s", b); }
    LOG("vk: H0 indexed draws with GPU-written arguments per frame %.1f: distinct index starts %.1f, distinct pools %.2f, index-pool window %.2f MB on average",
        g_h0_ipd / 600.0, g_h0_ipd_starts / 600.0, g_h0_ipd_pools / 600.0, g_h0_ipd ? g_h0_ipd_bytes / (double)g_h0_ipd / 1048576.0 : 0.0);
    { uint64_t fr = atomic_exchange(&g_h0_xr_fresh, 0), frb = atomic_exchange(&g_h0_xr_fresh_b, 0), kp = atomic_exchange(&g_h0_xr_kept, 0); uint64_t nk[4]; for (int i = 0; i < 4; i++) nk[i] = atomic_exchange(&g_h0_xr_nokeep[i], 0);
      uint64_t r0max = 0; for (int k = 0; k < NCTX; k++) if (g_ctx[k].ring0_size > r0max) r0max = g_ctx[k].ring0_size;
      LOG("vk: H0 rings per frame: extra rings fresh %.2f (%.1f MB, about %.0f pages first written), taken from the kept ones %.2f; freed and not kept: mapping unknown %.2f, keep pool full (8) %.2f, keep byte cap %.2f, memory tight %.2f; contexts used in the 600 frames %d of %d, their rings %llu MB in all, the largest %llu MB",
          fr / 600.0, frb / 600.0 / 1048576.0, frb / 600.0 / 4096.0, kp / 600.0, nk[0] / 600.0, nk[1] / 600.0, nk[2] / 600.0, nk[3] / 600.0, __builtin_popcount(g_h0_ctx_used), NCTX,
          (unsigned long long)(atomic_load(&g_ring0_bytes) >> 20), (unsigned long long)(r0max >> 20)); }
    { uint64_t hn[2], hc[2]; for (int i = 0; i < 2; i++) { hn[i] = atomic_exchange(&g_h0_gwh_ns[i], 0); hc[i] = atomic_exchange(&g_h0_gwh_n[i], 0); }
      LOG("vk: H0 GPU-written-range lock held per frame: by the recording thread's lookups %.2f ms (%.0f times), by the completion thread %.2f ms (%.1f times); entries walked per lookup %.1f",
          hn[0] / 600e6, hc[0] / 600.0, hn[1] / 600e6, hc[1] / 600.0, g_h0_look ? (double)g_h0_walk / g_h0_look : 0.0); g_h0_look = g_h0_walk = 0; }
    {   // this thread (the presenting one) from getrusage, the recording thread from /proc: faults, user / system time, run-delay
        static struct timeval u0, s0; static long mn0; static uint64_t rd0, rr0; static int rtid0;
        struct rusage ru; double ums = 0, sms = 0; long mn = 0;
        if (!getrusage(RUSAGE_THREAD, &ru)) { if (u0.tv_sec || s0.tv_sec || u0.tv_usec || s0.tv_usec) { ums = (ru.ru_utime.tv_sec - u0.tv_sec) * 1e3 + (ru.ru_utime.tv_usec - u0.tv_usec) / 1e3; sms = (ru.ru_stime.tv_sec - s0.tv_sec) * 1e3 + (ru.ru_stime.tv_usec - s0.tv_usec) / 1e3; mn = ru.ru_minflt - mn0; }
                                              u0 = ru.ru_utime; s0 = ru.ru_stime; mn0 = ru.ru_minflt; }
        extern _Atomic int g_rec_tid; int rt = atomic_load(&g_rec_tid); uint64_t run = 0, wait = 0; double rdms = -1, rrms = -1;
        if (rt) { char pth[64], buf[128]; snprintf(pth, sizeof pth, "/proc/self/task/%d/schedstat", rt); FILE *f = fopen(pth, "r");
                  if (f) { if (fgets(buf, sizeof buf, f) && sscanf(buf, "%llu %llu", (unsigned long long *)&run, (unsigned long long *)&wait) == 2) {
                               if (rtid0 == rt && rd0) { rdms = (wait - rd0) / 600e6; rrms = (run - rr0) / 600e6; } rd0 = wait; rr0 = run; rtid0 = rt; } fclose(f); } }
        LOG("vk: H0 threads per frame: this (presenting) thread %d %s the recording thread %d: minor faults %.0f, user %.2f ms, system %.2f ms; the recording thread on a CPU %.2f ms, runnable but waiting (run-delay) %.2f ms",
            (int)gettid(), (int)gettid() == rt ? "is" : "is not", rt, mn / 600.0, ums / 600.0, sms / 600.0, rrms, rdms);
    }
    memset(g_h0_cmp, 0, sizeof g_h0_cmp); memset(g_h0_cmp_why, 0, sizeof g_h0_cmp_why); g_h0_cmp_pass = 0; memset(g_h0_gen, 0, sizeof g_h0_gen); memset(g_h0_ci, 0, sizeof g_h0_ci);
    memset(g_h0_mrgen, 0, sizeof g_h0_mrgen); memset(g_h0_wscan, 0, sizeof g_h0_wscan); memset(g_h0_wscan_n, 0, sizeof g_h0_wscan_n); memset(g_h0_ic_rescan, 0, sizeof g_h0_ic_rescan);
    g_h0_ipd = g_h0_ipd_starts = g_h0_ipd_pools = g_h0_ipd_bytes = g_h0_ipd_copies = 0; g_h0_ctx_used = 0;
}
static void meas_lines(void) {
    { uint64_t n = atomic_exchange(&g_cpr_n, 0), ns = atomic_exchange(&g_cpr_ns, 0), nb = atomic_exchange(&g_cpr_bufs, 0), nr = atomic_exchange(&g_cpr_rec, 0);
      LOG("vk: command pools per frame over the last 600 (DM_VK_POOL_RESET %d): %.1f reset by dm-complete (%.3f ms, %.1f of them buffer by buffer), %.2f left to the recording thread; %d submission contexts",
          pool_reset_mode(), n / 600.0, ns / 600e6, nb / 600.0, nr / 600.0, NCTX); }
    { uint64_t cn = atomic_exchange(&g_cl_n, 0), cns = atomic_exchange(&g_cl_ns, 0), wn = atomic_exchange(&g_gw8_n, 0), wns = atomic_exchange(&g_gw8_ns, 0), lag = atomic_exchange(&g_gw8_lag, 0), pend = atomic_exchange(&g_gw8_pend, 0);
      LOG("vk: completion per frame over the last 600 (DM_VK_EARLY_DONE %d): a fence reached the game %.3f ms after the GPU ended it on average (%.1f submissions); the game waited for the GPU %.1f times, %.2f ms"
          " (per wait: %.2f submissions made after the one waited for, %.2f still running up to it); the GPU idle between game submissions %.2f ms a frame (%.1f gaps)",
          early_done_on(), cn ? cns / (double)cn / 1e6 : 0.0, cn / 600.0, wn / 600.0, wns / 600e6, wn ? lag / (double)wn : 0.0, wn ? pend / (double)wn : 0.0,
          atomic_exchange(&g_gpu_gap_ns, 0) / 600e6, atomic_exchange(&g_gpu_gap_n, 0) / 600.0); }
    uint64_t hit = 0, miss = 0, mb = 0; for (int i = 0; i < CI_N; i++) { hit += g_ci_hit[i]; miss += g_ci_miss[i]; mb += g_ci_mbytes[i]; }
    static const char *cn[CI_N] = {"vertex", "index", "index pool", "uniform", "storage", "texel", "image copy"};
    char cb[600]; size_t o = 0; cb[0] = 0;
    for (int i = 0; i < CI_N; i++) o += (size_t)snprintf(cb + o, sizeof cb - o, "%s%s %.1f/%.1f (%.2f MB)", i ? ", " : "", cn[i], g_ci_hit[i] / 600.0, g_ci_miss[i] / 600.0, g_ci_mbytes[i] / 600.0 / 1048576.0);
    LOG("vk: copy_in per frame over the last 600: %.1f reused, %.1f copied (%.2f MB); reused/copied by caller: %s; copies by reason: free slot %.1f, slot held by another range %.1f, earlier submission %.1f, larger %.1f,"
        " alignment %.1f, written since %.1f; by pool (estimate: one miss in 8 looked up): GPU-only %.1f, CPU cached %.1f, CPU uncached %.1f, unknown %.1f", hit / 600.0, miss / 600.0, mb / 600.0 / 1048576.0, cb,
        g_ci_why[MR_EMPTY] / 600.0, g_ci_why[MR_OTHER] / 600.0, g_ci_why[MR_GEN] / 600.0, g_ci_why[MR_SIZE] / 600.0, g_ci_why[MR_ALIGN] / 600.0, g_ci_why[MR_WLOG] / 600.0,
        g_ci_pool[PC_GPU] / 600.0, g_ci_pool[PC_CACHED] / 600.0, g_ci_pool[PC_UNCACHED] / 600.0, g_ci_pool[PC_NONE] / 600.0);
    uint64_t draws = g_tex_draws_win ? g_tex_draws_win : 1;
    LOG("vk: index ranges per frame over the last 600: %.1f reused, %.1f scanned (%.0f indices); scans of the same range again %.1f, over another live range %.1f, into a free or old slot %.1f;"
        " %llu distinct ranges in the 600 frames%s; texel buffer views made %.1f a frame; tex_get %.1f a draw", g_ic_hit / 600.0, g_ic_scan / 600.0, g_ic_idx / 600.0, g_ic_same / 600.0, g_ic_coll / 600.0, g_ic_new / 600.0,
        (unsigned long long)g_ic_keys, g_ic_keys_over ? " (the set was full: more)" : "", atomic_exchange(&g_bv_made, 0) / 600.0, g_texget_n / (double)draws); g_texget_n = 0;
    LOG("vk: CPU-written linear textures per frame over the last 600: %.1f content hashes (%.2f MB hashed), %.2f found changed and uploaded, %.1f of them in dirty-tracked storage; %.1f lookups left to the dirty pages (DM_VK_LCPU_DIRTY %d)", g_lcpu_hashes / 600.0, g_lcpu_bytes / 600.0 / 1048576.0, g_lcpu_changed / 600.0, g_lcpu_tracked / 600.0, g_lcpu_gated / 600.0, lcpu_dirty_on());
    g_lcpu_hashes = g_lcpu_bytes = g_lcpu_changed = g_lcpu_tracked = g_lcpu_gated = 0;
    LOG("vk: draws with a vertex attribute of format 0 read as zeros (DM_VK_ZERO_ATTR %d) per frame over the last 600: %.1f", zero_attr_on(), g_zattr_draws / 600.0); g_zattr_draws = 0;
    if (g_fbf_draws) { LOG("vk: DM_VK_FB_FAST per frame over the last 600: %.1f draws looked at, %.1f answered from the shader cache, %.1f by texture serials alone, %.1f with the full way for a matching serial%s",
                           g_fbf_draws / 600.0, g_fbf_cached / 600.0, g_fbf_serial / 600.0, g_fbf_full / 600.0, "");
                       if (g_fbf_chk) LOG("vk: DM_VK_FB_FAST=2 over the last 600 frames: %llu draws checked against the full way, %llu differed", (unsigned long long)g_fbf_chk, (unsigned long long)g_fbf_bad);
                       g_fbf_draws = g_fbf_cached = g_fbf_serial = g_fbf_full = g_fbf_chk = g_fbf_bad = 0; }
    if (g_neon_chk) { LOG("vk: DM_VK_IDX_NEON=2 over the last 600 frames: %llu scans checked against the scalar scan, %llu differed", (unsigned long long)g_neon_chk, (unsigned long long)g_neon_bad); g_neon_chk = g_neon_bad = 0; }
    if (g_shw_all[0] || g_shw_all[1])
        LOG("vk: DM_VK_SHADOW per frame over the last 600 (an estimate: a direct-mapped table of 64K, exact address and length only): copies whose bytes an earlier copy of the range had: same submission %.1f (%.2f MB), another submission of the frame %.1f (%.2f MB), an earlier frame %.1f (%.2f MB)"
            " of %.1f copies; index scans likewise: %.1f, %.1f, %.1f of %.1f (%.0f / %.0f / %.0f indices' bytes KB); hashing %.2f ms",
            g_shw_n[0][0] / 600.0, g_shw_b[0][0] / 600.0 / 1048576.0, g_shw_n[0][1] / 600.0, g_shw_b[0][1] / 600.0 / 1048576.0, g_shw_n[0][2] / 600.0, g_shw_b[0][2] / 600.0 / 1048576.0, g_shw_all[0] / 600.0,
            g_shw_n[1][0] / 600.0, g_shw_n[1][1] / 600.0, g_shw_n[1][2] / 600.0, g_shw_all[1] / 600.0, g_shw_b[1][0] / 600.0 / 1024.0, g_shw_b[1][1] / 600.0 / 1024.0, g_shw_b[1][2] / 600.0 / 1024.0, g_shw_ns / 600e6);
    if (g_shw_all[0]) { static const char *cn2[CI_N] = {"vertex", "index", "index pool", "uniform", "storage", "texel", "image copy"}; char b[600]; size_t o = 0; b[0] = 0;   // by caller (vk19.66m)
        for (int i = 0; i < CI_N; i++) if (g_shw_cb[i][0] || g_shw_cb[i][1]) o += (size_t)snprintf(b + o, sizeof b - o, "%s%s %.2f / %.2f MB", o ? ", " : "", cn2[i], g_shw_cb[i][0] / 600.0 / 1048576.0, g_shw_cb[i][1] / 600.0 / 1048576.0);
        LOG("vk: DM_VK_SHADOW copies' repeated bytes per frame by caller (another submission of the frame / an earlier frame): %s", o ? b : "none"); memset(g_shw_cb, 0, sizeof g_shw_cb); }
    { uint32_t v[600]; uint64_t k = g_npp < 600 ? g_npp : 600; memcpy(v, g_pp, k * sizeof *v); qsort(v, k, sizeof *v, u32cmp);
      uint64_t rc = 0; for (int i = 0; i < 8; i++) rc += g_rcore[i]; char cc[160]; size_t q = 0; cc[0] = 0;
      for (int i = 0; i < 8; i++) if (g_rcore[i]) q += (size_t)snprintf(cc + q, sizeof cc - q, "%scpu%d %.0f%%", q ? " " : "", i, 100.0 * g_rcore[i] / (rc ? rc : 1));
      int vk_stats_cluster_of(int cpu); uint64_t cn[8] = {0}; int best = -1;   // the cluster it ran on most (an A/B compares windows on the same one)
      for (int i = 0; i < 8; i++) { int f = vk_stats_cluster_of(i); if (f >= 0 && f < 8) cn[f] += g_rcore[i]; }
      for (int i = 0; i < 8; i++) if (cn[i] && (best < 0 || cn[i] > cn[best])) best = i;
      if (best >= 0) q += (size_t)snprintf(cc + q, sizeof cc - q, "; mostly on the cluster of cpu%d (%.0f%%)", best, 100.0 * cn[best] / (rc ? rc : 1));
      if (k) LOG("vk: present to present over the last %llu frames: p50 %.2f ms, p95 %.2f, p99 %.2f, longest %.2f; the recording thread at submission starts: %s", (unsigned long long)k,
                 v[k / 2] / 1e3, v[k * 95 / 100] / 1e3, v[k * 99 / 100] / 1e3, v[k - 1] / 1e3, cc);
      memset(g_rcore, 0, sizeof g_rcore); g_npp = 0; }
    { int tid[32]; uint64_t nf[32]; static int ptid[32]; static uint64_t pnf[32]; static int pk; int k = nvn_dirty_thread_faults(tid, nf, 32);
      uint64_t mc, mpg, mns; static uint64_t pmc, pmns; nvn_dirty_mp_stats(&mc, &mpg, &mns);
      char fb[700]; size_t q = 0; fb[0] = 0; uint64_t all = 0;
      for (int i = 0; i < k; i++) { uint64_t prev = 0; for (int j = 0; j < pk; j++) if (ptid[j] == tid[i]) prev = pnf[j]; uint64_t d = nf[i] - prev; all += d;
          if (d && q < sizeof fb - 40) { char p[64], nm[20] = "?"; snprintf(p, sizeof p, "/proc/self/task/%d/comm", tid[i]); FILE *f = fopen(p, "re"); if (f) { if (fgets(nm, sizeof nm, f)) nm[strcspn(nm, "\n")] = 0; fclose(f); }
                                        q += (size_t)snprintf(fb + q, sizeof fb - q, "%s%s (%d) %.2f", q ? ", " : "", nm, tid[i], d / 600.0); } }
      memcpy(ptid, tid, sizeof tid); memcpy(pnf, nf, sizeof nf); pk = k;
      LOG("vk: guest-memory protection per frame over the last 600: page faults %.2f (%s); mprotect %.1f calls, %.3f ms", all / 600.0, q ? fb : "none", (mc - pmc) / 600.0, (mns - pmns) / 600e6);
      pmc = mc; pmns = mns; }
    if (ts_any()) { int vk_stats_maps_last(void);   // DM_VK_TEXEL_STORE (vk19.70), DM_VK_BUF_STORE (vk19.72)
        int ne = 0; for (int i = 0; i < TS_NENT; i++) ne += g_ts[i].used;
        LOG("vk: range store per kind per frame over the last 600: texel %.1f served (%.2f MB), %.2f admitted; vertex %.1f (%.2f MB), %.2f; index pool %.2f (%.2f MB), %.2f",
            g_ts_khit[0] / 600.0, g_ts_khit_b[0] / 600.0 / 1048576.0, g_ts_kadm[0] / 600.0, g_ts_khit[1] / 600.0, g_ts_khit_b[1] / 600.0 / 1048576.0, g_ts_kadm[1] / 600.0,
            g_ts_khit[2] / 600.0, g_ts_khit_b[2] / 600.0 / 1048576.0, g_ts_kadm[2] / 600.0);
        memset(g_ts_khit, 0, sizeof g_ts_khit); memset(g_ts_khit_b, 0, sizeof g_ts_khit_b); memset(g_ts_kadm, 0, sizeof g_ts_kadm);
        LOG("vk: range store (DM_VK_TEXEL_STORE=%d) per frame over the last 600, all kinds: %.1f served (%.2f MB), %.1f missed, %.2f admitted, %.2f copied again, %.2f evicted, %.2f refused for a while (%.2f could not be admitted); views reused %.1f, made %.2f; verify %.1f checked, %llu differed; store %.1f of %.0f MB in %d blocks, %d waiting for readers, dropped %llu times; refused for the session %llu so far; mprotect failures %llu; memory map %d lines",
            ts_mode(), g_ts_hit / 600.0, g_ts_hit_b / 600.0 / 1048576.0, g_ts_miss / 600.0, g_ts_adm / 600.0, g_ts_recopy / 600.0, g_ts_evict / 600.0, g_ts_bl / 600.0, g_ts_refused / 600.0, g_ts_vreuse / 600.0, g_ts_vmade / 600.0,
            g_ts_vchk / 600.0, (unsigned long long)g_ts_vbad, g_ts_inuse * (double)TS_GRAN / 1048576.0, g_ts_ngran * (double)TS_GRAN / 1048576.0, ne, g_ts_nret, (unsigned long long)g_ts_drops, (unsigned long long)g_tsbl_n,
            (unsigned long long)nvn_dirty_mp_failures(), vk_stats_maps_last());
        g_ts_hit = g_ts_hit_b = g_ts_miss = g_ts_adm = g_ts_recopy = g_ts_evict = g_ts_bl = g_ts_vmade = g_ts_vreuse = g_ts_vchk = g_ts_refused = 0; }
    { void dm_guest_sync_report(char *out, size_t n, uint64_t frames); static char gs[8192]; dm_guest_sync_report(gs, sizeof gs, 600);   // hle_thread.c (vk19.59b)
      LOG("vk: guest sync per frame over the last 600, the 8 busiest threads: %s", gs); }
    memset(g_ci_hit, 0, sizeof g_ci_hit); memset(g_ci_miss, 0, sizeof g_ci_miss); memset(g_ci_mbytes, 0, sizeof g_ci_mbytes); memset(g_ci_why, 0, sizeof g_ci_why); memset(g_ci_pool, 0, sizeof g_ci_pool);
    g_ic_hit = g_ic_scan = g_ic_idx = g_ic_same = g_ic_coll = g_ic_new = g_ic_keys = g_ic_keys_over = 0; memset(g_ick, 0, sizeof g_ick); g_tex_draws_win = 0;
    memset(g_shw_n, 0, sizeof g_shw_n); memset(g_shw_b, 0, sizeof g_shw_b); memset(g_shw_all, 0, sizeof g_shw_all); g_shw_ns = 0;
}
uint64_t mtl_present(const tex_t *t, double period_s, const int crop[4]) {
    if (g_ntz) texu_zero_recheck();
    (void)period_s;
    uint64_t n = atomic_fetch_add(&g_presents, 1) + 1;
    g_frame_draw = 0; vk_ctl(n);
    { extern _Atomic uint64_t dm_present_ns; atomic_store(&dm_present_ns, now_ns()); }   // hle_thread.c: DM_HITCH_DUMP measures stalls from it (nvn_mtl.m sets it on Apple; never set here before, so the dump never fired on Android)
    { uint64_t now = now_ns(); if (g_pp_last) g_pp[g_npp++ % 600] = (uint32_t)((now - g_pp_last) / 1000 > UINT32_MAX ? UINT32_MAX : (now - g_pp_last) / 1000); g_pp_last = now; }   // present to present (us)
    if (n == 1 || n % 600 == 0) { void vk_stats_window(uint64_t frames); vk_stats_window(n); }   // heat and energy (vk_stats.c): started at the first frame, a window each 600
    {   // the frame's work for the hint session: the time since the last present (its CPU side) and the GPU time per frame of the last completed submissions
        static uint64_t lastp, gprev; uint64_t now = now_ns(), g = atomic_load(&g_gpu_cum), gd = g >= gprev ? g - gprev : 0; gprev = g;
        if (lastp && now - lastp < 1000000000ull) adpf_frame(now - lastp, gd < 200000000ull ? gd : 0);
        lastp = now; }
    {   // a long frame (over 50 ms from the last present): what it held -- new pipelines, draws left out for theirs, waits for a context and for GPU-written data, page faults
        // (major ones: read back from swap or a file), extra rings, descriptor pools, uploads, GPU time. At most 20 lines every 600 frames
        static uint64_t lp, c_pso, c_pson, c_skip, c_wait, c_up, c_xr, c_dp, c_gww, c_gpu, win, c_tcpu, c_rec; static long c_mj, c_mn; static int nl;
        uint64_t now = now_ns(); struct rusage ru, rt; getrusage(RUSAGE_SELF, &ru); getrusage(RUSAGE_THREAD, &rt);
        uint64_t tcpu = (uint64_t)(rt.ru_utime.tv_sec + rt.ru_stime.tv_sec) * 1000000000ull + (uint64_t)(rt.ru_utime.tv_usec + rt.ru_stime.tv_usec) * 1000ull, rec = atomic_load(&g_rec_ns);
        uint64_t pso = (uint64_t)g_npso, pson = atomic_load(&g_pso_ns), skip = atomic_load(&g_pso_skipped), wt = atomic_load(&g_wait_cum), up = atomic_load(&g_up_cum),
                 xr = atomic_load(&g_xring_n), dp = (uint64_t)atomic_load(&g_ndp_total), gww = atomic_load(&g_gw_wait_cum), gpu = atomic_load(&g_gpu_cum);
        if (n / 600 != win) { win = n / 600; nl = 0; }
        if (lp && now - lp > 50000000ull && nl < 20) { nl++;
            LOG("vk: long frame %llu: %.1f ms (this thread on the CPU %.1f ms, recording %.1f ms); new pipelines %llu (%.1f ms), draws left out %llu; waiting for a context %.1f ms, for GPU-written data %.1f ms; page faults %ld major, %ld minor; extra rings %llu, descriptor pools %llu; uploads %.1f MB; GPU %.1f ms",
                (unsigned long long)n, (now - lp) / 1e6, (tcpu - c_tcpu) / 1e6, rec >= c_rec ? (rec - c_rec) / 1e6 : 0.0, (unsigned long long)(pso - c_pso), (pson - c_pson) / 1e6, (unsigned long long)(skip - c_skip), (wt - c_wait) / 1e6, (gww - c_gww) / 1e6,
                ru.ru_majflt - c_mj, ru.ru_minflt - c_mn, (unsigned long long)(xr - c_xr), (unsigned long long)(dp - c_dp), (up - c_up) / 1048576.0, (gpu - c_gpu) / 1e6); }
        lp = now; c_tcpu = tcpu; c_rec = rec; c_pso = pso; c_pson = pson; c_skip = skip; c_wait = wt; c_up = up; c_xr = xr; c_dp = dp; c_gww = gww; c_gpu = gpu; c_mj = ru.ru_majflt; c_mn = ru.ru_minflt;
    }
    if (n == 2) { pthread_t t; if (!pthread_create(&t, NULL, pso_precreate, NULL)) pthread_detach(t); }   // the pipelines earlier runs made, again, behind the game's first frames   // draws are numbered from 0 in each frame; the live control file (vk_ctl)
    {   // pacing for DM_FPS=30 (every frame two vsyncs) and the frame-rate counter's reading
        static uint64_t last, t0, n0; uint64_t now = now_ns();
        uint64_t per = atomic_load_explicit(&g_pace_fifo, memory_order_relaxed) ? 33333333ull - 33333333ull / 32 : 33333333ull;   // FIFO paces (vk19.63c): the host 1/32 short
        if (mtl_fps_target() == 30 && last && now - last < per) { usleep((useconds_t)((per - (now - last)) / 1000)); now = now_ns(); }
        last = now; if (!t0) { t0 = now; n0 = n; } else if (now - t0 >= 1000000000ull) { g_fps_shown = (double)(n - n0) * 1e9 / (double)(now - t0); t0 = now; n0 = n; }
    }
    if (t && g_has_surface && !g_tl) mtl_finish();   // no timeline semaphore: the frame's work is done before its present reads the window texture
    if (t && g_has_surface) win_present(t, crop);
    if (n == 1 || n % 600 == 0) LOG("vk: %llu frames presented, %llu draws, %llu dropped (no program %llu, shader %llu, no target %llu, indirect %llu, primitive %llu, pipeline %llu, descriptors %llu), %d pipelines, %d shaders",
                                    (unsigned long long)n, (unsigned long long)g_stat_draws, (unsigned long long)g_stat_skip, (unsigned long long)g_skip_why[0], (unsigned long long)g_skip_why[1], (unsigned long long)g_skip_why[2],
                                    (unsigned long long)g_skip_why[3], (unsigned long long)g_skip_why[4], (unsigned long long)g_skip_why[5], (unsigned long long)g_skip_why[6], g_npso, g_nsh);
    if (n % 600 == 0 && getenv("DM_VK_TEXCOUNT")) {   // diagnostics: live textures by kind, and how many share their storage start with an older one
        int ng = 0, nu = 0, nn = 0, dup = 0; lk_(&g_rmu, LW_RMU);
        for (int h = 0; h < NHASH; h++) for (vtex *v = g_tex[h]; v; v = v->next) {
            if (v->gpu) ng++; else if (v->uploaded) nu++; else nn++;
            for (int h2 = 0; h2 < NHASH && v->mem; h2++) for (vtex *o = g_tex[h2]; o; o = o->next) if (o != v && o->mem == v->mem && o->serial < v->serial && o->w == v->w && o->nvn_fmt == v->nvn_fmt) { dup++; h2 = NHASH; break; }
        }
        pthread_mutex_unlock(&g_rmu);
        LOG("vk: live textures: %d rendered to, %d uploaded, %d neither; %d on the storage of an older one of the same size and format", ng, nu, nn, dup);
    }
    if (n % 600 == 0) { uint64_t live = 0, held = 0; int nb = 0; lk_(&g_blkmu, LW_BLKMU); for (int i = 0; i < g_nblk; i++) if (g_blk[i].m) { nb++; held += g_blk[i].size; live += g_blk[i].live; } pthread_mutex_unlock(&g_blkmu);
                        LOG("vk: image memory: %d blocks of 64 MB (%.0f MB, %llu live images in them), big images %.0f MB now; rings %.0f MB extra", nb, held / 1048576.0, (unsigned long long)live, atomic_load(&g_big_bytes) / 1048576.0, atomic_load(&g_extra_ring_bytes) / 1048576.0); }
    if (n % 600 == 0 && (g_fb_copies || g_fb_chain || g_fb_asis)) LOG("vk: same-pass feedback snapshots so far: %llu; mip-chain draws through a view cut below the level drawn: %llu; layered or base-level ones left as they are: %llu", (unsigned long long)g_fb_copies, (unsigned long long)g_fb_chain, (unsigned long long)g_fb_asis);
    if (n % 600 == 0) LOG("vk: texture slots read as black so far: no handle %llu, handle not registered %llu, texture not created %llu, no view %llu; lookups of a dead texture object %llu (registrations of one %llu)", (unsigned long long)g_tex_miss[0],
                          (unsigned long long)g_tex_miss[1], (unsigned long long)g_tex_miss[2], (unsigned long long)g_tex_miss[3], (unsigned long long)g_tex_dead, (unsigned long long)g_treg_dead);
    if (n % 600 == 0) { extern uint64_t g_nsub_stat; static uint64_t ls;
        LOG("vk: recording per frame over the last 600: draws %.2f ms (descriptor sets %.2f ms, %.0f%% reused), dispatches %.2f ms, texture uploads %.2f ms (%.1f MB), vkQueueSubmit %.2f ms; %.1f submissions a frame;"
            " GPU-written data of a running submission used in place %llu times, copied %llu times, waited for %llu times (%.2f ms a frame; indirect draws %llu, partial ranges %llu, misfits in-submission %llu); copies from GPU-written bytes: into textures %llu, buffers %llu; dispatch write-backs %.1f KB of %.1f KB looked at; reads of an earlier submission's GPU-written range: its bytes as they are %llu, with the CPU's rewrites since patched in %llu; texture copies from guest memory over a dispatch's range %llu; those compares and patches %.2f ms (%.1f MB compared; patches %.2f ms, %.1f a frame, %.0f copy regions a frame); held-pass sampling checks %.2f ms; vertex index bias pushed %.1f draws a frame (%.1f not 0); GPU-written range lookups %.0f a frame, %.0f answered at once, %.0f entries looked at",
            g_t_draw / 600e6, g_t_ws / 600e6, g_dsc_hit + g_dsc_miss ? 100.0 * g_dsc_hit / (g_dsc_hit + g_dsc_miss) : 0.0, g_t_disp / 600e6, g_t_up / 600e6, g_up_bytes / 600e6 * 1e3 / 1e3, g_t_sub / 600e6,
            (g_nsub_stat - ls) / 600.0, (unsigned long long)atomic_exchange(&g_gw_held, 0), (unsigned long long)atomic_exchange(&g_gw_copied, 0), (unsigned long long)atomic_exchange(&g_gw_waits, 0), atomic_exchange(&g_gw_wait_ns, 0) / 600e6,
            (unsigned long long)atomic_exchange(&g_gw_why[0], 0), (unsigned long long)atomic_exchange(&g_gw_why[1], 0), (unsigned long long)atomic_exchange(&g_gw_why[2], 0),
            (unsigned long long)atomic_exchange(&g_gw_b2t, 0), (unsigned long long)atomic_exchange(&g_gw_b2b, 0), g_wb_words * 4 / 1024.0, g_wb_bytes / 1024.0,
            (unsigned long long)atomic_exchange(&g_gw_gpu_reads, 0), (unsigned long long)atomic_exchange(&g_gw_cpu_fix, 0), (unsigned long long)atomic_exchange(&g_gw_b2t_guest, 0),
            g_t_stale / 600e6, g_stale_bytes / 600.0 / 1048576.0, g_t_patch / 600e6, g_npatch / 600.0, g_patch_regions / 600.0, g_t_split / 600e6, g_vid_push / 600.0, g_vid_nz / 600.0, g_gw_calls / 600.0, g_gw_quick / 600.0, g_gw_walked / 600.0);
        g_wb_words = g_wb_bytes = 0; g_t_stale = g_stale_bytes = 0; g_t_patch = g_patch_regions = g_npatch = 0; g_vid_push = g_vid_nz = 0; g_t_split = 0; g_gw_calls = g_gw_quick = g_gw_walked = 0;
        { double miss_us = g_dsc_nmiss_t ? g_dsc_miss_ns / 1e3 / g_dsc_nmiss_t : 0;
          LOG("vk: descriptor sets per frame over the last 600 (DM_VK_DSET_CACHE2 %d): %.1f reused, %.1f made (%.1f us each to allocate and write), %.2f rejected on the full compare; ring blocks kept by content %.1f, put anew %.1f; the reuse saved about %.2f ms a frame; texture storage scans %.1f a frame, %.1f left out as uploaded this frame (DM_VK_DIRTY_ONCE %d); driver segments taken as built %.1f, built %.1f (DM_VK_SEG_CACHE %d); stale patches by dispatch %.2f (%.0f runs); held reads %.1f, their writer %.1f submissions back on average, %llu at most; forwarded %.2f (%.2f MB) (DM_VK_GW_FORWARD %d); entries ended at their write-back %.1f",
              dset_cache2(), g_dsc_hit / 600.0, g_dsc_miss / 600.0, miss_us, g_dsc_rej / 600.0, g_memo_hit / 600.0, g_memo_miss / 600.0, g_dsc_hit * miss_us / 1e3 / 600.0, g_dirty_scans / 600.0, g_dirty_skips / 600.0, dirty_once(), g_segc_hit / 600.0, g_segc_miss / 600.0, seg_cache_on(), g_patch_disp / 600.0, g_patch_runs / 600.0, g_hold_n / 600.0, g_hold_n ? (double)g_hold_age_sum / g_hold_n : 0.0, (unsigned long long)g_hold_age_max, g_fwd_n / 600.0, g_fwd_bytes / 600.0 / 1048576.0, gw_forward_k(), atomic_exchange(&g_gw_wb_ended, 0) / 600.0);
          g_memo_hit = g_memo_miss = g_dsc_rej = g_dsc_miss_ns = g_dsc_nmiss_t = 0; g_dirty_scans = g_dirty_skips = 0; g_segc_hit = g_segc_miss = 0; g_patch_disp = g_patch_runs = 0; g_hold_n = g_hold_age_sum = g_hold_age_max = 0; g_fwd_n = g_fwd_bytes = 0;
          if (htab_on()) { LOG("vk: handle table per frame over the last 600 (DM_VK_HTAB): texture slots %.1f from the table, %.1f looked up (first in the submission or a registration since), %.1f gone the full way (the image needed tex_get)",
              g_ht_hit / 600.0, g_ht_miss / 600.0, g_ht_slow / 600.0); g_ht_hit = g_ht_miss = g_ht_slow = 0; }
          LOG("vk: stage-level set reuse per frame over the last 600 (DM_VK_STAGE_REUSE %d): %.1f bound again unresolved, %.1f resolved (%.1f of them not reusable: storage images %.1f, storage texel buffers %.1f, other texel buffers %.1f, written or odd storage buffers %.1f, handles in uniforms %.1f, other uniforms %.1f, too many %.1f)",
              stage_reuse(), g_wsc_hit / 600.0, g_wsc_miss / 600.0, g_wsc_nofit / 600.0, g_wsc_nf[WSC_NF_SIMG] / 600.0, g_wsc_nf[WSC_NF_STB] / 600.0, g_wsc_nf[WSC_NF_TB] / 600.0, g_wsc_nf[WSC_NF_SSBO] / 600.0,
              g_wsc_nf[WSC_NF_TEXU] / 600.0, g_wsc_nf[WSC_NF_UBO] / 600.0, g_wsc_nf[WSC_NF_MANY] / 600.0);
          g_wsc_hit = g_wsc_miss = g_wsc_nofit = 0; memset(g_wsc_nf, 0, sizeof g_wsc_nf);
          if (zc_mode()) { LOG("vk: zero-copy pools: %.0f MB backed; per frame %.1f reads taken in place, %.1f copied because a command-list write overlapped", g_zc_bytes / 1048576.0, g_zc_hits / 600.0, g_zc_miss_wlog / 600.0); g_zc_hits = g_zc_miss_wlog = 0; }
          if (store_none_on()) { LOG("vk: DM_VK_STORE_NONE per frame over the last 600: %.1f passes through secondaries; colour attachments stored %.1f, left unstored %.1f; depth stored %.1f, left unstored %.1f",
              g_sn_passes / 600.0, g_sn_all_c / 600.0, g_sn_none_c / 600.0, g_sn_all_d / 600.0, g_sn_none_d / 600.0); g_sn_passes = g_sn_all_c = g_sn_none_c = g_sn_all_d = g_sn_none_d = 0; }
          LOG("vk: state per frame over the last 600 (DM_VK_STATE_SKIP %d): pipelines taken as the last draw's %.1f, state commands left out %.1f, binds left out %.1f", state_skip(), g_pso_memo / 600.0, g_state_skips / 600.0, g_bind_skips / 600.0); g_pso_memo = g_state_skips = g_bind_skips = 0;
          static const char *const lwn[LW_N] = {"textures (rmu)", "GPU-written ranges (gwmu)", "contexts (mu)", "queue (qmu)", "image blocks (blkmu)"}; char lb[512]; int lo = 0;
          for (int k = 0; k < LW_N; k++) lo += snprintf(lb + lo, sizeof lb - lo, "%s%s %.2f ms (%.1f)", k ? ", " : "", lwn[k], atomic_exchange(&g_lw_ns[k], 0) / 600e6, atomic_exchange(&g_lw_n[k], 0) / 600.0);
          LOG("vk: lock waits per frame over the last 600, all threads: %s", lb);
          if (patch_verify()) LOG("vk: stale patch check (DM_VK_PATCH_VERIFY): %llu dispatches; right after the dispatch %llu words checked, %llu differ; at completion %llu checked, %llu differ", (unsigned long long)g_pv_checked,
                                  (unsigned long long)g_pv_snap_words, (unsigned long long)g_pv_snap_bad, (unsigned long long)g_pv_words, (unsigned long long)g_pv_bad);
          LOG("vk: fresh ring memory per frame over the last 600: extra rings %.2f (%.1f MB; %llu taken from the kept ones in all), context rings made again %.3f (%.1f MB)",
              g_xr_fresh / 600.0, g_xr_fresh_bytes / 600.0 / 1048576.0, (unsigned long long)g_xc_reused, g_ring_new / 600.0, g_ring_new_bytes / 600.0 / 1048576.0);
          g_xr_fresh = g_xr_fresh_bytes = g_ring_new = g_ring_new_bytes = 0; }
        ls = g_nsub_stat; g_t_draw = g_t_ws = g_t_disp = g_t_up = g_t_sub = 0; g_up_bytes = 0; g_dsc_hit = g_dsc_miss = 0; }
    if (n % 600 == 0) {   // memory: the process (RSS, its anonymous part, PSS) and what the backend holds
        long rss = 0, anon = 0, pss = 0; char ln[256]; FILE *f = fopen("/proc/self/status", "re");
        if (f) { while (fgets(ln, sizeof ln, f)) { sscanf(ln, "VmRSS: %ld", &rss); sscanf(ln, "RssAnon: %ld", &anon); } fclose(f); }
        static long pss_last; if (n % 6000 == 0 && (f = fopen("/proc/self/smaps_rollup", "re"))) { while (fgets(ln, sizeof ln, f)) if (sscanf(ln, "Pss: %ld", &pss_last) == 1) break; fclose(f); }   // every 6000 frames: smaps_rollup walks every mapping of the process (the fixed hibernation layout's tens of GB), a 70 ms frame each time it was read every 600
        pss = pss_last;
        int nb = 0; lk_(&g_blkmu, LW_BLKMU); for (int i = 0; i < g_nblk; i++) if (g_blk[i].m) nb++; pthread_mutex_unlock(&g_blkmu);
        long cached = 0, avail = 0; if ((f = fopen("/proc/meminfo", "re"))) { while (fgets(ln, sizeof ln, f)) { sscanf(ln, "Cached: %ld", &cached); sscanf(ln, "MemAvailable: %ld", &avail); } fclose(f); }
        extern _Atomic uint64_t g_fs_dropped;
        LOG("vk: memory: process RSS %.0f MB (anonymous %.0f MB), PSS %.0f MB; Vulkan rings %.0f MB + extra %.0f MB, image blocks %d x 64 MB (new ones made on the recording thread %llu, spares made %llu / used %llu / freed by trims %llu / not needed once made %llu), big images %.0f MB, %d descriptor pools;"
            " system: available %.0f MB, page cache %.0f MB; game file reads dropped from the page cache so far %.0f MB; textures evicted %llu, made again %llu; drawn before their data came and filled when it did %llu of %llu; GPU copies written into storage too: buffer %llu, texture %llu",
            rss / 1024.0, anon / 1024.0, pss / 1024.0, atomic_load(&g_ring0_bytes) / 1048576.0, atomic_load(&g_extra_ring_bytes) / 1048576.0, nb, (unsigned long long)g_blk_new_rec, (unsigned long long)g_blk_spare_made, (unsigned long long)g_blk_spare_used, (unsigned long long)g_blk_spare_freed, (unsigned long long)g_blk_spare_dropped, atomic_load(&g_big_bytes) / 1048576.0, atomic_load(&g_ndp_total),
            avail / 1024.0, cached / 1024.0, atomic_load(&g_fs_dropped) / 1048576.0, (unsigned long long)g_evicted, (unsigned long long)g_remade, (unsigned long long)g_late_fill, (unsigned long long)g_late_watch, (unsigned long long)g_mir_b2t, (unsigned long long)g_mir_t2t);
    }
    if (n % 600 == 0) { static uint64_t pr, prs, pd, pq; LOG("vk: occlusion per frame over the last 600 (DM_VK_OCC_MODE %d%s): %.1f samples-passed reports, %.1f resets, %.1f draws counted in %.1f queries", occ_mode(), g_occ_cum ? ", counting everything" : "",
                          (g_occ_nrep - pr) / 600.0, (g_occ_nrst - prs) / 600.0, (g_occ_ndraw - pd) / 600.0, (g_occ_nq - pq) / 600.0); pr = g_occ_nrep; prs = g_occ_nrst; pd = g_occ_ndraw; pq = g_occ_nq; }
    if (n % 600 == 0) LOG("vk: texel buffer slots read as zeros so far: nothing bound %llu, not a buffer texture %llu, format not usable %llu", (unsigned long long)g_tbuf_miss[0],
                          (unsigned long long)g_tbuf_miss[1], (unsigned long long)g_tbuf_miss[2]);
    if (n % 600 == 0) { uint64_t gp = atomic_exchange(&g_gpu_kind_ns[0], 0), gd = atomic_exchange(&g_gpu_kind_ns[1], 0), gc = atomic_exchange(&g_gpu_kind_ns[2], 0);
                        LOG("vk: GPU per frame over the last 600: render passes %.2f ms, dispatches %.2f ms, copies %.2f ms, the upload command buffers %.2f ms (the rest of the submissions' span is time between them); %.1f render passes a frame (%.1f of them clears only; colour targets set in another view format %.1f), SetRenderTargets detours held %.1f and same targets kept %.1f a frame, held passes ended for a draw that samples an attachment %.1f a frame (left open %.1f: the attachment it samples was not written in the pass)",
                            gp / 600e6, gd / 600e6, gc / 600e6, atomic_exchange(&g_up_gpu_ns, 0) / 600e6, g_npass / 600.0, g_npass_clr / 600.0, g_rt_fmt_differs / 600.0, g_nheld / 600.0, g_nsame / 600.0, g_nsplit / 600.0, g_nsplit_skip / 600.0); g_npass = 0; g_npass_clr = 0; g_rt_fmt_differs = 0; g_nheld = 0; g_nsame = 0; g_nsplit = 0; g_nsplit_skip = 0; }
    if (n % 600 == 0) gsig_report(600);
    if (n % 600 == 0) {   // the host heap (malloc) and what the backend keeps in it, to attribute the process's native heap (dumpsys meminfo)
        struct mallinfo mi = mallinfo(); int nt = 0, nv = 0, live = 0; uint64_t snap = 0, xc = 0;
        lk_(&g_rmu, LW_RMU); for (int hh = 0; hh < NHASH; hh++) for (vtex *v = g_tex[hh]; v; v = v->next) { nt++; nv += v->nviews; } pthread_mutex_unlock(&g_rmu);
        pthread_mutex_lock(&g_psomu); for (uint32_t i = 0; i < g_psot_cap; i++) if (g_psot[i] && atomic_load(&g_psot[i]->state) == 1 && g_psot[i]->p) live++; pthread_mutex_unlock(&g_psomu);
        pthread_mutex_lock(&g_snapmu); snap = g_snap_kept; pthread_mutex_unlock(&g_snapmu); pthread_mutex_lock(&g_xcmu); xc = g_xc_bytes; pthread_mutex_unlock(&g_xcmu);
        LOG("vk: host heap: %.0f MB in use (malloc's own count, %.0f MB mapped); the backend: %d textures (%d views), %d pipelines alive of %d known, %d shaders, snapshot pool %.0f MB, extra rings kept %.0f MB",
            (unsigned)mi.uordblks / 1048576.0, ((unsigned)mi.arena + (unsigned)mi.hblkhd) / 1048576.0, nt, nv, live, g_npso, g_nsh, snap / 1048576.0, xc / 1048576.0);
    }
    if (n % 600 == 0) { static long mn0, mj0; struct rusage ru; if (!getrusage(RUSAGE_THREAD, &ru)) { if (mn0 || mj0) LOG("vk: this thread's page faults over the last 600 frames: %.1f minor, %.2f major a frame", (ru.ru_minflt - mn0) / 600.0, (ru.ru_majflt - mj0) / 600.0); mn0 = ru.ru_minflt; mj0 = ru.ru_majflt; } }
    if (n % 600 == 0) clock_line();
    if (n % 600 == 0) { LOG("vk: ADPF over the last 600 frames: %llu reports%s, %llu failed (last error %d), reported CPU %.1f ms and GPU %.1f ms a frame on average (target 16.7 ms)",
                            (unsigned long long)g_adpf_n, g_adpf_v2 ? " with the GPU's work" : "", (unsigned long long)g_adpf_fail, g_adpf_err, g_adpf_n ? g_adpf_cpu / 1e6 / g_adpf_n : 0.0, g_adpf_n ? g_adpf_gpu / 1e6 / g_adpf_n : 0.0);
                        g_adpf_n = g_adpf_fail = g_adpf_cpu = g_adpf_gpu = 0; }
    if (n % 600 == 0 && defer_on()) { LOG("vk: DM_VK_DISPATCH_DEFER: %.1f dispatches a frame recorded after their pass, %.1f passes a frame ended early for a draw reading what one writes, %.1f barriers a frame that kept their pass open", g_ndefer_n / 600.0, g_ndefer_haz / 600.0, g_nbar_kept / 600.0); g_ndefer_n = g_ndefer_haz = 0; g_nbar_kept = 0; }
    if (n % 600 == 0 && (ubwc_on() || pass_barrier_on())) { LOG("vk: DM_VK_UBWC %d: %llu images made with a format list or not mutable so far, %llu views outside the list, copies into them: %llu from buffers, %llu from images, %llu uploads; DM_VK_PASS_BARRIER %d: %.1f pass ends a frame without vertex shaders, %.1f vertex-shader barriers a frame",
        ubwc_on(), (unsigned long long)g_ubwc_imgs, (unsigned long long)g_ubwc_fallback, (unsigned long long)g_ubwc_b2t, (unsigned long long)g_ubwc_t2t, (unsigned long long)g_ubwc_up, pass_barrier_on(), g_nbar_narrow / 600.0, g_nbar_vs / 600.0); g_nbar_narrow = g_nbar_vs = 0; }
    if (n % 600 == 0) { LOG("vk: waiting for a free submission context per frame: %.2f ms for its own submission, %.2f ms for a later one still reading its ring (%d contexts)",
                        atomic_exchange(&g_wait_fence_ns, 0) / 600e6, atomic_exchange(&g_wait_hold_ns, 0) / 600e6, NCTX);
                        if (ctx_pick_on()) { uint64_t tot = g_pick_free + g_pick_grow + g_pick_wait;
                            LOG("vk: DM_VK_CTX_PICK per frame over the last 600: %.1f submissions, %.1f took a free context at once (%.0f%%), %.2f made a new one, %.2f waited at the cap (%.2f for its own submission, %.2f for a later reader); contexts %d, at most %d",
                                tot / 600.0, g_pick_free / 600.0, tot ? 100.0 * g_pick_free / tot : 0.0, g_pick_grow / 600.0, g_pick_wait / 600.0, g_pick_wait_fence / 600.0, g_pick_wait_hold / 600.0, NCTX, g_pick_hw);
                            g_pick_free = g_pick_grow = g_pick_wait = g_pick_wait_fence = g_pick_wait_hold = 0; } }
    if (n % 600 == 0) { LOG("vk: per frame over the last 600: GPU %.2f ms, recording %.2f ms, waiting for the GPU %.2f ms, present waiting %.2f ms", atomic_exchange(&g_gpu_ns, 0) / 600e6,
                            atomic_exchange(&g_rec_ns, 0) / 600e6, atomic_exchange(&g_wait_ns, 0) / 600e6, atomic_exchange(&g_pwait_ns, 0) / 600e6); }
    h0_tick(n);
    if (n % 600 == 0) { h0_lines(); meas_lines(); }
    if (t) shot(t, n, crop);
    // On demand: DM_VK_DUMP_FILE=<path>; when that file appears (checked every 30 frames) the next frame is pass-logged and its render targets written, and the file is removed
    // (a capture where the frame number cannot be known in advance: wherever the player happens to be)
    static uint64_t trig; static const char *tf = (const char *)1; if (tf == (const char *)1) tf = getenv("DM_VK_DUMP_FILE");
    if (trig && n == trig) { rt_dump(n, 1); trig = 0; } else rt_dump(n, 0);
    if (tf && !trig && n % 30 == 0 && !access(tf, F_OK)) { trig = n + 1; unlink(tf); LOG("vk: %s found: frame %llu is logged and its render targets written", tf, (unsigned long long)trig); }
    if (n % 1800 == 0 || (g_pso_last_new && n - g_pso_last_new == 120 && !atomic_load(&g_pso_pending))) pcache_save();   // also soon after new pipelines settle
    if (n % 600 == 0) LOG("vk: pipelines: %d made, %llu on worker threads, %llu made again from earlier runs, %llu draws left out while theirs was made, %.0f ms making them, %.0f ms loading shaders, in all; warm ones made again at first use %llu (%.0f ms); lookups so far: found ready %llu, found warm %llu, found being made %llu, not found %llu", g_npso,
                          (unsigned long long)atomic_load(&g_pso_async), (unsigned long long)atomic_load(&g_pre_made), (unsigned long long)atomic_load(&g_pso_skipped), atomic_load(&g_pso_ns) / 1e6, atomic_load(&g_sh_ns) / 1e6,
                          (unsigned long long)atomic_load(&g_pso_rewarm), atomic_load(&g_pso_rewarm_ns) / 1e6,
                          (unsigned long long)g_pso_look[0], (unsigned long long)g_pso_look[1], (unsigned long long)g_pso_look[2], (unsigned long long)g_pso_look[3]);
    { static int64_t pl = -2; static int on; if (pl == -2) { const char *e = getenv("DM_VK_PASSLOG"); pl = e ? atoll(e) : -1; on = e && strchr(e, '+'); }   // <frame>+: every frame from there on
      g_plog = (on ? (int64_t)n + 1 >= pl && pl >= 0 : (int64_t)n + 1 == pl) || (trig && n + 1 == trig); if (g_plog && on) LOG("plog: frame %llu", (unsigned long long)n + 1); }
    return n;
}
void mtl_present_wait(uint64_t ticket) { (void)ticket; }
double mtl_presented_time(uint64_t n) { (void)n; return 0; }
int mtl_has_window(void) { return 0; }
int mtl_display_hz(void) { return 0; }
int mtl_fps_target(void) { static int f = -1; if (f < 0) { const char *e = getenv("DM_FPS"); f = e && atoi(e) == 30 ? 30 : 60; } return f; }
int mtl_link_active(void) { return 0; }
int mtl_link_wait_tick(int vsyncs, uint64_t timeout_ns) { (void)vsyncs; (void)timeout_ns; return 0; }
uint32_t mtl_present_qos(void) { return 0; }
void mtl_ub_present(double iv_ms) { (void)iv_ms; }
void mtl_set_depth_zero(int on) {   // vertex shaders loaded from here on follow it (spv_gl_depth); the games set it once at device init, before their first shader
    if (on != g_depth_zero && g_nsh && g_prewarm_depth < 0) LOG("[RED] vk: depth mode changed to %s after %d shaders were loaded: those keep the other convention", on ? "near = 0" : "GL (near = -w)", g_nsh);
    if (g_prewarm_depth >= 0 && on != g_prewarm_depth && g_nsh) { static int n; if (on && !n++) LOG("[RED] vk: the game's depth mode (near = 0) differs from the one the shader pre-warm used"); }
    g_depth_zero = on;
    // kept for the next run's pre-warm (dm_vk_prewarm loads shaders before the game sets it): the last mode the game set
    const char *d = getenv("DM_VK_DIR"); if (d) { char p[1100]; snprintf(p, sizeof p, "%s/../vk_depth_mode", d); FILE *f = fopen(p, "w"); if (f) { fprintf(f, "%d\n", on); fclose(f); } }
}
void mtl_set_origin_lower_left(int on) { g_origin_ll = on; }
void mtl_set_frame(uint64_t f) { (void)f; }
void mtl_xv_window_tex(uint64_t serial) { (void)serial; }
void mtl_read_texels(const tex_t *t, unsigned level, unsigned layer0, const NVNcopyRegion *r, uint8_t *dst, size_t row, size_t img) {
    (void)t; (void)level; (void)layer0; (void)row; RED_ONCE(80, "CPU texture reads return zeros");
    memset(dst, 0, img * (size_t)(r->depth > 0 ? r->depth : 1));
}

// ---------- Zero-copy pools (DM_VK_ZEROCOPY, vk19.46, opt-in) ----------
// As the Metal host does with newBufferWithBytesNoCopy: a game memory pool's pages are backed by an AHardwareBuffer (a dmabuf), mapped over the pool's own address range
// (MAP_SHARED | MAP_FIXED, the bytes copied out and back) when the pool is made, and imported into Vulkan as one buffer, so the GPU reads the guest's bytes where they are.
// Stage Z0 (=1): vertex and index streams bind the pool buffer at their offset instead of copying into the ring, unless a dispatch wrote them (gw_find first) or a command-list
// write of this generation overlaps them (the write log). Pools: not CPU_NO_ACCESS (0x1), PHYSICAL (0x100) or VIRTUAL (0x200), 64 KB to 256 MB, DM_VK_ZEROCOPY_MB (384) in all
// (dmabuf memory cannot be swapped). A pool's release puts anonymous memory back at its address with the same bytes; the AHB and its buffer go when the submissions recorded
// before the release are done. Probed on the Adreno 825 (ahbprobe): the GPU sees the CPU's writes and the CPU the GPU's, without an invalidate; mprotect on the mapping faults
#include <android/hardware_buffer.h>
#include <sys/mman.h>
#include <sys/socket.h>
static int zc_mode(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_ZEROCOPY"); v = e ? atoi(e) : 0; if (v < 0) v = 0; if (v) LOG("vk: DM_VK_ZEROCOPY=%d", v); } return v; }
typedef struct { uint64_t lo, hi; const void *mem; VkBuffer b; VkDeviceMemory m; AHardwareBuffer *ahb; uint64_t dead; } zcpool;
enum { NZC = 256 };
static zcpool g_zc[NZC]; static int g_nzc; static pthread_mutex_t g_zcmu = PTHREAD_MUTEX_INITIALIZER;   // g_zc_dev: the device has the AHB import enabled; g_zc_bytes / hits / miss_wlog (declared at the top)
static struct { const void *mem; int flags; } g_zc_pf[16]; static int g_zc_npf;   // flags of the pool being made (dm_vk_pool_flags, then mtl_pool_add)
void dm_vk_pool_flags(const void *mem, int flags) { pthread_mutex_lock(&g_zcmu); g_zc_pf[g_zc_npf++ % 16] = (typeof(g_zc_pf[0])){mem, flags}; pthread_mutex_unlock(&g_zcmu); }
static int recv_fd(int sock) { char buf[256]; struct iovec io = {buf, sizeof buf}; char c[CMSG_SPACE(sizeof(int) * 4)]; struct msghdr m = {0}; m.msg_iov = &io; m.msg_iovlen = 1; m.msg_control = c; m.msg_controllen = sizeof c;
    if (recvmsg(sock, &m, 0) < 0) return -1; for (struct cmsghdr *h = CMSG_FIRSTHDR(&m); h; h = CMSG_NXTHDR(&m, h)) if (h->cmsg_type == SCM_RIGHTS) { int fd; memcpy(&fd, CMSG_DATA(h), sizeof fd); return fd; } return -1; }
static void vk_init(void);
static void zc_back(const void *mem, uint64_t size) {
    int flags = -1; pthread_mutex_lock(&g_zcmu); for (int i = 0; i < 16; i++) if (g_zc_pf[i].mem == mem) { flags = g_zc_pf[i].flags; g_zc_pf[i].mem = NULL; } pthread_mutex_unlock(&g_zcmu);
    if (flags < 0 || (flags & 0x301) || size < (64u << 10) || size > (256u << 20)) return;
    static uint64_t cap; if (!cap) { const char *e = getenv("DM_VK_ZEROCOPY_MB"); cap = (uint64_t)(e ? atol(e) : 384) << 20; }
    uint64_t lo = (uint64_t)(uintptr_t)mem & ~4095ull, hi = ((uint64_t)(uintptr_t)mem + size + 4095) & ~4095ull, n = hi - lo;
    vk_init(); if (!g_zc_dev) { static int once; if (!once++) LOG("[RED] vk: zero-copy pools: the device has no AHardwareBuffer import"); return; }
    pthread_mutex_lock(&g_zcmu);
    int ok = g_nzc < NZC && g_zc_bytes + n <= cap;
    for (int i = 0; i < g_nzc && ok; i++) if (!g_zc[i].dead && lo < g_zc[i].hi && g_zc[i].lo < hi) ok = 0;   // a page shared with a backed pool: left as it is
    if (!ok) { pthread_mutex_unlock(&g_zcmu); return; }
    AHardwareBuffer_Desc d = {.width = (uint32_t)n, .height = 1, .layers = 1, .format = AHARDWAREBUFFER_FORMAT_BLOB,
        .usage = AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN | AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN | AHARDWAREBUFFER_USAGE_GPU_DATA_BUFFER};
    AHardwareBuffer *ahb = NULL; int fd = -1, sv[2] = {-1, -1}; uint8_t *save = NULL; VkBuffer b = VK_NULL_HANDLE; VkDeviceMemory m = VK_NULL_HANDLE; const char *why = NULL;
    if (AHardwareBuffer_allocate(&d, &ahb)) why = "AHardwareBuffer_allocate";
    else if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) || AHardwareBuffer_sendHandleToUnixSocket(ahb, sv[0]) || (fd = recv_fd(sv[1])) < 0) why = "the dmabuf fd";
    if (!why) {   // the Vulkan buffer first: nothing is remapped unless it imports
        static PFN_vkGetAndroidHardwareBufferPropertiesANDROID gp; if (!gp) gp = (PFN_vkGetAndroidHardwareBufferPropertiesANDROID)vkGetDeviceProcAddr(g_dev, "vkGetAndroidHardwareBufferPropertiesANDROID");
        VkAndroidHardwareBufferPropertiesANDROID ap = {VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID};
        VkExternalMemoryBufferCreateInfo eb = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO, NULL, VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID};
        VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &eb, 0, n, ALL_BUF_USAGE, VK_SHARING_MODE_EXCLUSIVE, 0, NULL};
        if (!gp || gp(g_dev, ahb, &ap) != VK_SUCCESS) why = "the AHB's properties";
        else if (vkCreateBuffer(g_dev, &bi, NULL, &b) != VK_SUCCESS) why = "vkCreateBuffer";
        else {
            uint32_t mt = 0; while (mt < 32 && !(ap.memoryTypeBits >> mt & 1)) mt++;
            VkMemoryDedicatedAllocateInfo ded = {VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, NULL, VK_NULL_HANDLE, b};
            VkImportAndroidHardwareBufferInfoANDROID imp = {VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID, &ded, ahb};
            VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &imp, ap.allocationSize, mt};
            if (vkAllocateMemory(g_dev, &ai, NULL, &m) != VK_SUCCESS) { imp.pNext = NULL; if (vkAllocateMemory(g_dev, &ai, NULL, &m) != VK_SUCCESS) { m = VK_NULL_HANDLE; why = "the import"; } }
            if (m && vkBindBufferMemory(g_dev, b, m, 0) != VK_SUCCESS) why = "vkBindBufferMemory";
        }
    }
    if (!why && !(save = malloc(n))) why = "a copy buffer";
    if (!why) {
        memcpy(save, (void *)(uintptr_t)lo, n);
        if (mmap((void *)(uintptr_t)lo, n, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd, 0) != (void *)(uintptr_t)lo) { why = "mmap MAP_FIXED"; mmap((void *)(uintptr_t)lo, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_FIXED | MAP_ANONYMOUS, -1, 0); }
        memcpy((void *)(uintptr_t)lo, save, n);   // the pool's bytes, in the new pages (or back in anonymous ones)
    }
    free(save); if (sv[0] >= 0) { close(sv[0]); close(sv[1]); } if (fd >= 0) close(fd);   // the mapping keeps the dmabuf
    if (why) { if (b) vkDestroyBuffer(g_dev, b, NULL); if (m) vkFreeMemory(g_dev, m, NULL); if (ahb) AHardwareBuffer_release(ahb); pthread_mutex_unlock(&g_zcmu);
               static int nl; if (nl++ < 8) LOG("[RED] vk: zero-copy pool %p (%llu KB) not backed: %s failed", mem, (unsigned long long)(n >> 10), why); return; }
    g_zc[g_nzc++] = (zcpool){lo, hi, mem, b, m, ahb, 0}; g_zc_bytes += n;
    pthread_mutex_unlock(&g_zcmu);
    static int nl; if (nl++ < 64) LOG("vk: zero-copy pool %p: %llu KB backed by an AHardwareBuffer (flags %#x; %llu MB in all)", mem, (unsigned long long)(n >> 10), flags, (unsigned long long)(g_zc_bytes >> 20));
}
static void zc_unback(const void *mem) {
    pthread_mutex_lock(&g_zcmu);
    for (int i = 0; i < g_nzc; i++) if (g_zc[i].mem == mem && !g_zc[i].dead) {
        uint64_t n = g_zc[i].hi - g_zc[i].lo; uint8_t *save = malloc(n);
        if (save) { memcpy(save, (void *)(uintptr_t)g_zc[i].lo, n); mmap((void *)(uintptr_t)g_zc[i].lo, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_FIXED | MAP_ANONYMOUS, -1, 0); memcpy((void *)(uintptr_t)g_zc[i].lo, save, n); free(save); }
        g_zc[i].dead = g_fence + 2;   // after the submission being recorded, too
        g_zc_bytes -= n;
    }
    pthread_mutex_unlock(&g_zcmu);
}
static void zc_reap(uint64_t done) {   // the released pools' buffers and AHBs, once nothing recorded before the release runs
    pthread_mutex_lock(&g_zcmu);
    for (int i = 0; i < g_nzc;) if (g_zc[i].dead && g_zc[i].dead <= done) { vkDestroyBuffer(g_dev, g_zc[i].b, NULL); vkFreeMemory(g_dev, g_zc[i].m, NULL); AHardwareBuffer_release(g_zc[i].ahb); g_zc[i] = g_zc[--g_nzc]; } else i++;
    pthread_mutex_unlock(&g_zcmu);
}
// The pool buffer and offset holding [addr, addr + n) whole, when it is backed and no command-list write of this generation touches it (those are applied to guest memory as the
// list is walked: a draw recorded before one must not see it); 0 otherwise
static int zc_find(uint64_t addr, uint64_t n, VkBuffer *b, VkDeviceSize *off) {
    if (!g_nzc || !n) return 0;
    int hit = 0; pthread_mutex_lock(&g_zcmu);
    for (int i = 0; i < g_nzc; i++) if (!g_zc[i].dead && addr >= g_zc[i].lo && addr + n <= g_zc[i].hi) { *b = g_zc[i].b; *off = addr - g_zc[i].lo; hit = 1; break; }
    pthread_mutex_unlock(&g_zcmu);
    if (!hit) return 0;
    for (uint32_t k = 0; k < g_nwlog; k++) if (g_wlo[k] < addr + n && g_whi[k] > addr) { g_zc_miss_wlog++; return 0; }
    g_zc_hits++; return 1;
}
// Memory pools: nothing to do (draws copy what they read); mtl_addr answers for the NVN layer whether an address is in one
#define NPOOL 4096
static struct { const uint8_t *mem; uint64_t size; int cls; } g_pool[NPOOL];   // cls: the pool's CPU access (PC_*, for the copy counters), from its flags (dm_vk_pool_flags)
static int g_npool;
static pthread_mutex_t g_pool_mu = PTHREAD_MUTEX_INITIALIZER;
void mtl_pool_add(const void *mem, uint64_t size) {
    int flags = -1; pthread_mutex_lock(&g_zcmu); for (int i = 0; i < 16; i++) if (g_zc_pf[i].mem == mem) flags = g_zc_pf[i].flags; pthread_mutex_unlock(&g_zcmu);   // read before zc_back takes them
    int cls = flags < 0 ? PC_NONE : (flags & NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS) ? PC_GPU : (flags & NVN_MEMORY_POOL_FLAGS_CPU_UNCACHED) ? PC_UNCACHED : PC_CACHED;
    if (zc_mode()) zc_back(mem, size);
    pthread_mutex_lock(&g_pool_mu); if (g_npool < NPOOL) { g_pool[g_npool].mem = mem; g_pool[g_npool].size = size; g_pool[g_npool].cls = cls; g_npool++; } pthread_mutex_unlock(&g_pool_mu);
}
static int pool_class(uint64_t addr) {   // copy_in misses (the submit thread): the last pool found is tried first, without the lock
    static uint64_t lb, le; static int lc = PC_NONE;
    if (addr >= lb && addr < le) return lc;
    int c = PC_NONE; pthread_mutex_lock(&g_pool_mu);
    for (int i = 0; i < g_npool; i++) { uint64_t b = (uint64_t)(uintptr_t)g_pool[i].mem; if (addr >= b && addr < b + g_pool[i].size) { c = g_pool[i].cls; lb = b; le = b + g_pool[i].size; lc = c; break; } }
    pthread_mutex_unlock(&g_pool_mu); return c;
}
static uint64_t pool_range(uint64_t addr, uint64_t *size) {   // the guest address of the memory pool holding addr and its size; 0 when none
    uint64_t r = 0; *size = 0; pthread_mutex_lock(&g_pool_mu);
    for (int i = 0; i < g_npool; i++) { uint64_t b = (uint64_t)(uintptr_t)g_pool[i].mem; if (addr >= b && addr < b + g_pool[i].size) { r = b; *size = g_pool[i].size; break; } }
    pthread_mutex_unlock(&g_pool_mu); return r;
}
static uint64_t pool_base(uint64_t addr) {   // the guest address of the memory pool holding addr, 0 when none (H0)
    uint64_t r = 0; pthread_mutex_lock(&g_pool_mu);
    for (int i = 0; i < g_npool; i++) { uint64_t b = (uint64_t)(uintptr_t)g_pool[i].mem; if (addr >= b && addr < b + g_pool[i].size) { r = b; break; } }
    pthread_mutex_unlock(&g_pool_mu); return r;
}
static uint64_t pool_left(uint64_t addr, uint64_t cap) {
    uint64_t n = 0; pthread_mutex_lock(&g_pool_mu);
    for (int i = 0; i < g_npool; i++) { uint64_t b = (uint64_t)(uintptr_t)g_pool[i].mem; if (addr >= b && addr < b + g_pool[i].size) { n = b + g_pool[i].size - addr; break; } }
    pthread_mutex_unlock(&g_pool_mu); return n < cap ? n : cap;
}
void mtl_pool_del(const void *mem) { if (zc_mode()) zc_unback(mem); pthread_mutex_lock(&g_pool_mu); for (int i = 0; i < g_npool; i++) if (g_pool[i].mem == mem) { if (g_ts_buf) ts_pool_deleted(mem, g_pool[i].size); g_pool[i] = g_pool[--g_npool]; break; } pthread_mutex_unlock(&g_pool_mu); }
int mtl_addr(uint64_t addr, uint64_t len, void **buf, uint64_t *off) {
    int hit = 0; pthread_mutex_lock(&g_pool_mu);
    for (int i = 0; i < g_npool && !hit; i++) { uint64_t b = (uint64_t)(uintptr_t)g_pool[i].mem; if (addr >= b && addr + len <= b + g_pool[i].size) { if (buf) *buf = (void *)g_pool[i].mem; if (off) *off = addr - b; hit = 1; } }
    pthread_mutex_unlock(&g_pool_mu); return hit;
}
void mtl_alias_drop_pool(const void *pool) { (void)pool; }

// statistics the NVN layer prints
void mtl_stats(uint64_t *passes, uint64_t *draws) { *passes = 0; *draws = g_stat_draws; }
void mtl_vbuf_stats(uint64_t *resolve, uint64_t *reuse, uint64_t *set, uint64_t *skip, uint64_t *copies, uint64_t *copy_bytes) { *resolve = *reuse = *set = *skip = *copies = *copy_bytes = 0; }
void mtl_use_stats(uint64_t *call, uint64_t *skip) { *call = *skip = 0; }
uint64_t mtl_upload_ns_take(void) { return 0; }
uint64_t mtl_upload_gpu_ns_take(void) { return 0; }
uint64_t mtl_upload_bytes(void) { return 0; }
uint64_t mtl_udc_saved_bytes(void) { return 0; }
void mtl_ubp_why(uint64_t w[5]) { memset(w, 0, 5 * sizeof *w); }
void mtl_ubp_stats(uint64_t *a, uint64_t *b, uint64_t *c) { *a = *b = *c = 0; }
void mtl_ubd_stats(uint64_t *a, uint64_t *b) { *a = *b = 0; }
void mtl_res_stats(char *out, size_t n) { if (n) snprintf(out, n, "vulkan"); }
int mtl_pd_on(void) { return 0; }   // the Metal host's pass dump (queue events in it, hle_nvn.c): none here
void mtl_pd_note(const char *what, uint64_t v) { (void)what; (void)v; }
void mtl_pkg_stats(uint64_t *uploads, uint64_t *reuploads) { *uploads = g_pkg_up; *reuploads = g_pkg_reup; }
uint64_t mtl_pkg_miss(void) { return 0; }
void mtl_mem_stat(unsigned long long a, unsigned long long b) { (void)a; (void)b; }
void mtl_lc_stats(uint64_t *a, uint64_t *b) { *a = *b = 0; }
int mtl_gpu_span(uint64_t f, uint64_t *t0, uint64_t *t1) { (void)f; (void)t0; (void)t1; return 0; }
double mtl_gpu_ms(uint64_t f) { (void)f; return 0; }
uint64_t mtl_ds_omit(void) { return 0; }
uint64_t mtl_drawable_ns_take(void) { return 0; }
uint64_t mtl_cut_count(void) { return 0; }
uint64_t mtl_blit_splits(void) { return 0; }
void mtl_create_stats(uint64_t *lib, uint64_t *pso) { *lib = 0; *pso = (uint64_t)g_npso; }
void mtl_create_ns(uint64_t *lib_ns, uint64_t *pso_ns) { *lib_ns = atomic_load(&g_sh_ns); *pso_ns = atomic_load(&g_pso_ns); }   // shader loads, pipeline creation (all threads)
void mtl_bw(uint64_t *load, uint64_t *store, uint64_t *full) { *load = *store = *full = 0; }
void mtl_barrier_stats(uint64_t *skip, uint64_t *full) { *skip = *full = 0; }
void mtl_args_stats(uint64_t *enc, uint64_t *reuse) { *enc = *reuse = 0; }
void mtl_alloc_stats(uint64_t *w, uint64_t *g, int *n) { *w = *g = 0; *n = 0; }
// Entry points the NVN layer gained after this backend was written (nvn_mtl.h / hle_nvn.c), answered the Vulkan way: images register into the texture table as before
// (RegisterImage), a queue's captured window origin is the one the replay follows from its submission on, the rest are Metal-only features (present snapshots, the display link,
// the pipeline archive, frame captures) that are off here, and statistics of work this backend does not do
void mtl_reg_image(int id, const tex_t *t, const view_t *v) { mtl_reg_tex(id, t, v); }
void mtl_set_queue_origin_lower_left(int on) { g_origin_ll = on; }
int mtl_wait_acq(uint64_t fence, uint64_t timeout_ns) { return mtl_wait(fence, timeout_ns); }
void mtl_queue_activate(uint64_t serial) { (void)serial; }
void mtl_queue_forget(uint64_t serial) { (void)serial; }
void mtl_prepare(void) {}
int mtl_snap_on(void) { return 0; }
uint64_t mtl_snap_acquire_wait(void) { return 0; }
int mtl_link_wait_ticks(int ticks, uint64_t timeout_ns) { (void)ticks; (void)timeout_ns; return 0; }
int mtl_link_tick(uint64_t n, double *target, double *lead_ms, int *ticks, double *gap_ms) { (void)n; (void)target; (void)lead_ms; (void)ticks; (void)gap_ms; return 0; }
int mtl_link_stalled(void) { return 0; }
int mtl_link_rate(void) { return 60; }
int mtl_pso_prebuild_tool(void) { return 0; }
void mtl_pso_line(char *out, size_t n) { if (n) snprintf(out, n, "vulkan: %d made", g_npso); }
void nvn_cap_submit(int n, const uint64_t *h) { (void)n; (void)h; }
void nvn_cap_present(const tex_t *t, const int crop[4]) { (void)t; (void)crop; }
void dm_gfx_shot_at(long k, const char *path) { (void)k; (void)path; LOG("[RED] vk: DM_CTL shot is Metal-only (DM_VK_SHOTS writes frames here)"); }
void dm_gfx_texraw(uint64_t serial, const char *path) { (void)serial; (void)path; }
void dm_gfx_texlist(int target) { (void)target; }
void dm_gfx_texdump_next(const char *path) { (void)path; }
void dm_gfx_passdump_next(long n) { (void)n; }
void mtl_ab_gpu(uint64_t f, double gpu_ms, double frame_ms) { (void)f; (void)gpu_ms; (void)frame_ms; }
unsigned mtl_cpu_tex_count(void) { return (unsigned)g_ncpu; }
uint64_t mtl_pkg_edge_skips(void) { return 0; }
uint64_t mtl_copy_cut_count(void) { return 0; }
void mtl_sm_stats(uint64_t *a, uint64_t *b, uint64_t *c) { *a = *b = *c = 0; }
void mtl_hybrid_stats(uint64_t *draws, uint64_t *made, uint64_t *none) { *draws = *made = *none = 0; }
void mtl_geomesh_stats(uint64_t *a, uint64_t *b) { *a = *b = 0; }
void mtl_geohoist_stats(uint64_t *a, uint64_t *b, uint64_t *c) { *a = *b = 0; memset(c, 0, 6 * sizeof *c); }
void mtl_cshoist_stats(uint64_t *a, uint64_t *b, uint64_t *c) { *a = *b = 0; memset(c, 0, 8 * sizeof *c); }

// ---------- Hibernation (dm/host/hle_hibsnap.c): this backend's lasting state in a snapshot ----------
// The images, views, samplers and pipelines are caches filled from guest memory as draws need them. What the game set up through NVN calls is kept: the memory pools, the texture
// and sampler descriptor tables, the textures the CPU writes, the depth and origin modes, and the fence counter (sync objects in guest memory hold fence values: every one up to the
// snapshot's is complete in the restored process). GPU-only content: before the snapshot, every registered texture the GPU filled (copies, render passes) is read back into its
// storage in guest memory and listed as written by the CPU, so the restored process uploads it from there (mtl_hib_readback, as nvn_mtl.m does; FINAL FANTASY XII THE ZODIAC AGE fills
// its UI atlases by GPU copies). Single-sampled colour textures only, and none the host decodes for want of the format (their image is not the storage's format)
typedef struct { uint32_t npool, ntreg, nsreg, ncpu; int32_t origin_ll, depth_zero; uint64_t fence; } hib_vk;
typedef struct { uint64_t mem, size; } hib_vpool;
typedef struct { int32_t idx, has_v; tex_t t; view_t v; } hib_vtreg;
typedef struct { int32_t idx, pad; samp_par p; } hib_vsreg;
static uint64_t *g_hib_rb; static uint32_t g_hib_nrb;
void mtl_hib_readback(void) {   // inside a submission (hle_nvn.c dm_hib_nvn_prepare), every guest thread parked and the GPU drained
    free(g_hib_rb); g_hib_rb = NULL; g_hib_nrb = 0;
    pthread_mutex_lock(&g_rmu);
    tex_t *ts = malloc(((size_t)g_ntreg + 1) * sizeof *ts); int nt = 0;
    for (int i = 0; i < g_ntreg; i++) if (g_treg[i].ok && g_treg[i].t.serial) ts[nt++] = g_treg[i].t;
    pthread_mutex_unlock(&g_rmu);
    typedef struct { tex_t t; unsigned lv, ly; uint32_t w, h, dz; size_t row, img; VkDeviceSize off; } rb_t;   // dz: a 3D level's slices (one copy of the whole level)
    rb_t *rb = malloc(((size_t)nt * 16 + 1) * sizeof *rb); int nrb = 0; VkDeviceSize total = 0;
    g_hib_rb = malloc(((size_t)nt + 1) * sizeof *g_hib_rb);
    for (int i = 0; i < nt; i++) {
        const tex_t *t = &ts[i]; int dup = 0;
        for (uint32_t k = 0; k < g_hib_nrb && !dup; k++) dup = g_hib_rb[k] == t->serial;
        if (dup || !t->mem || !t->pool || (t->pool->flags & NVN_MEMORY_POOL_FLAGS_VIRTUAL) || t->d.samples > 1 || t->d.target == NVN_TEXTURE_TARGET_BUFFER
            || (t->d.flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET)) || t->has_layout) continue;
        pthread_mutex_lock(&g_rmu); vtex *v = tex_find(t->serial); int cpu = cpu_written(t->serial); pthread_mutex_unlock(&g_rmu);
        if (!v || v->dead || !v->img || v->evicted || cpu ||   // released: kept findable until freed (mtl_tex_release), its storage may be another's now
            v->f.dec || v->f.depth || !(v->aspect & VK_IMAGE_ASPECT_COLOR_BIT) || (v->uploaded && !v->gpu && !v->copied && !(v->gpu_lv & ~v->mir_lv))) continue;   // copied levels whose bytes reached the storage too (mir_lv) need nothing   // not made, uploaded from its storage, or not its storage's format
        int bw = 1, bh = 1, bpb = nvn_fmt_bpb(t->d.format, &bw, &bh); if (!bpb) continue;
        for (unsigned lv = 0; lv < v->levels && lv < 16; lv++) {
            uint32_t w = v->w >> lv ? v->w >> lv : 1, h = v->h >> lv ? v->h >> lv : 1, dz = t->d.target == NVN_TEXTURE_TARGET_3D ? (v->d >> lv ? v->d >> lv : 1) : 1;
            size_t row = (size_t)((w + bw - 1) / bw) * bpb, img = row * ((h + bh - 1) / bh);
            for (unsigned ly = 0; ly < v->layers; ly++) {
                rb = realloc(rb, ((size_t)nrb + 1) * sizeof *rb);
                rb[nrb++] = (rb_t){*t, lv, ly, w, h, dz, row, img, total}; total += (img * dz + 15) & ~(VkDeviceSize)15;
            }
        }
        g_hib_rb[g_hib_nrb++] = t->serial;
    }
    if (nrb) {
        VkDeviceMemory m; void *map = NULL; VkBuffer b = buf_new(total, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &m, &map);
        VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT};
        pass_end();
        vkCmdPipelineBarrier(C->cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
        for (int i = 0; i < nrb; i++) {
            pthread_mutex_lock(&g_rmu); vtex *v = tex_find(rb[i].t.serial); pthread_mutex_unlock(&g_rmu);
            VkBufferImageCopy c = {rb[i].off, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, rb[i].lv, rb[i].ly, 1}, {0, 0, 0}, {rb[i].w, rb[i].h, rb[i].dz}};
            vkCmdCopyImageToBuffer(C->cb, v->img, VK_IMAGE_LAYOUT_GENERAL, b, 1, &c);
        }
        C->used = 1;
        uint64_t f = mtl_sub_commit(); mtl_wait(f, UINT64_MAX);   // the copies done: their bytes into the textures' storage
        for (int i = 0; i < nrb; i++) {
            NVNcopyRegion r = {0, 0, 0, (int)rb[i].w, (int)rb[i].h, (int)rb[i].dz};
            if (nvn_tex_mirror_b2t(&rb[i].t, rb[i].lv, rb[i].ly, &r, (const uint8_t *)map + rb[i].off, rb[i].row, rb[i].img)) { static int n; if (n++ < 4) LOG("[RED] hibernate: texture ser=%llu level %u not written into its storage", (unsigned long long)rb[i].t.serial, rb[i].lv); }
        }
        vkDestroyBuffer(g_dev, b, NULL); vkFreeMemory(g_dev, m, NULL);
        mtl_sub_begin();   // the caller commits a submission of its own
    }
    LOG("hibernate: %u GPU-filled textures read back into their storage (%.1f MB)", g_hib_nrb, total / 1048576.0);
    free(rb); free(ts);
}
void nvn_dirty_add_range(const void *mem, size_t size);
size_t mtl_hib_save(uint8_t **out) {
    hib_vk h = {0};
    pthread_mutex_lock(&g_pool_mu); h.npool = (uint32_t)g_npool; pthread_mutex_unlock(&g_pool_mu);
    pthread_mutex_lock(&g_rmu); for (int i = 0; i < g_ntreg; i++) h.ntreg += g_treg[i].ok != 0; for (int i = 0; i < g_nsreg; i++) h.nsreg += g_sreg_ok[i] != 0; h.ncpu = (uint32_t)g_ncpu + g_hib_nrb; pthread_mutex_unlock(&g_rmu);
    h.origin_ll = g_origin_ll; h.depth_zero = g_depth_zero; pthread_mutex_lock(&g_mu); h.fence = g_fence; pthread_mutex_unlock(&g_mu);
    size_t n = sizeof h + h.npool * sizeof(hib_vpool) + h.ntreg * sizeof(hib_vtreg) + h.nsreg * sizeof(hib_vsreg) + h.ncpu * 8;
    uint8_t *b = calloc(1, n), *p = b; memcpy(p, &h, sizeof h); p += sizeof h;
    pthread_mutex_lock(&g_pool_mu); for (uint32_t i = 0; i < h.npool; i++) { hib_vpool e = {(uint64_t)(uintptr_t)g_pool[i].mem, g_pool[i].size}; memcpy(p, &e, sizeof e); p += sizeof e; } pthread_mutex_unlock(&g_pool_mu);
    pthread_mutex_lock(&g_rmu);
    for (int i = 0; i < g_ntreg; i++) if (g_treg[i].ok) { hib_vtreg e = {i, g_treg[i].has_v, g_treg[i].t, g_treg[i].v}; memcpy(p, &e, sizeof e); p += sizeof e; }
    for (int i = 0; i < g_nsreg; i++) if (g_sreg_ok[i]) { hib_vsreg e = {i, 0, g_sreg[i]}; memcpy(p, &e, sizeof e); p += sizeof e; }
    memcpy(p, g_cpu_written, (size_t)g_ncpu * 8); p += (size_t)g_ncpu * 8;
    pthread_mutex_unlock(&g_rmu);
    memcpy(p, g_hib_rb, (size_t)g_hib_nrb * 8); free(g_hib_rb); g_hib_rb = NULL; g_hib_nrb = 0;   // read back for the snapshot: uploaded from their storage when restored
    *out = b; return n;
}
int mtl_hib_restore(const uint8_t *b, size_t n) {
    hib_vk h; if (n < sizeof h) return -1; memcpy(&h, b, sizeof h);
    if (n != sizeof h + h.npool * sizeof(hib_vpool) + h.ntreg * sizeof(hib_vtreg) + h.nsreg * sizeof(hib_vsreg) + h.ncpu * 8) return -1;
    const uint8_t *p = b + sizeof h;
    for (uint32_t i = 0; i < h.npool; i++) { hib_vpool e; memcpy(&e, p, sizeof e); p += sizeof e; mtl_pool_add((const void *)(uintptr_t)e.mem, e.size); nvn_dirty_add_range((const void *)(uintptr_t)e.mem, e.size); }
    for (uint32_t i = 0; i < h.ntreg; i++) { hib_vtreg e; memcpy(&e, p, sizeof e); p += sizeof e; mtl_reg_tex(e.idx, &e.t, e.has_v ? &e.v : NULL); }
    for (uint32_t i = 0; i < h.nsreg; i++) { hib_vsreg e; memcpy(&e, p, sizeof e); p += sizeof e; mtl_reg_samp(e.idx, &e.p); }
    pthread_mutex_lock(&g_rmu);
    for (uint32_t i = 0; i < h.ncpu; i++) { uint64_t s; memcpy(&s, p, 8); p += 8;
        if (!cpu_written(s)) { if (g_ncpu == g_ccpu) { g_ccpu = g_ccpu ? 2 * g_ccpu : 64; g_cpu_written = realloc(g_cpu_written, g_ccpu * sizeof *g_cpu_written); } g_cpu_written[g_ncpu++] = s; } }
    pthread_mutex_unlock(&g_rmu);
    g_origin_ll = h.origin_ll; g_depth_zero = h.depth_zero;
    pthread_mutex_lock(&g_mu); g_fence = g_done = h.fence; for (int k = 0; k < NCTX_MAX; k++) g_ctx[k].resetting = 0; pthread_cond_broadcast(&g_cv); pthread_mutex_unlock(&g_mu);   // (no reset is running after a restore: none half done)   // every fence the snapshot's syncs hold is complete
    LOG("hibernate: Vulkan replay state restored: %u pools, %u texture and %u sampler descriptors, %u CPU-written textures, fence %llu", h.npool, h.ntreg, h.nsreg, h.ncpu, (unsigned long long)h.fence);
    return 0;
}
