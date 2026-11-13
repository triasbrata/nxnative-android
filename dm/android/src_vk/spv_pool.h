// The pools of a shader whose texture handle the translator traced to no slot (dmsh_build.py pool_arrays, kinds tex_pool / samp_pool): it indexes the game's texture pool by the
// handle's texture ID and the sampler pool by its sampler ID. dmsh_xlate declares them as arrays of 1048576 textures and 4096 samplers in set 0 and clamps the index to them
// (UMin). spv_pool_patch moves them into the shared pool set (nvn_vk.c pool_on: set SPV_POOL_SET, a binding per texture class, SPV_POOL_SAMP for the samplers), sizes the texture
// arrays to the host's capacity cap (and clamps the index to cap - 1 where the access chain takes it), and marks the index, the element pointer, the loaded descriptor and the
// sampled image NonUniform (the ID can differ between invocations; SampledImageArrayNonUniformIndexing). No Vulkan here: dm/android/tests can run it over a game's SPIR-V on
// the Mac and check the result with spirv-val
#ifndef SPV_POOL_H
#define SPV_POOL_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define SPV_POOL_SET 3
enum { SPV_PC_2D, SPV_PC_2DA, SPV_PC_3D, SPV_PC_CUBE, SPV_PC_N };   // texture classes (float sampled type, single-sampled): their bindings in the pool set
#define SPV_POOL_SAMP SPV_PC_N                                      // the sampler pool's binding
#define SPV_POOL_SAMP_CAP 4096                                      // dmsh_xlate's sampler pool length, kept

typedef struct { uint32_t binding; int samp; int cls; } spv_pool_var;   // in: the binding (set 0) and whether it is the sampler pool; out: the texture class (-1 sampler)

// One instruction inserted before instruction index at (or appended to a section), as words
typedef struct { size_t at; uint32_t w[8]; int n; } spv_ins;

static const char *spv_pool_err;   // why the last spv_pool_patch failed

// Whether opcode op is a value instruction with a result type at word 1 and a result at word 2 (the ones an index can come from; anything else fails the patch)
static int spv_value_op(uint32_t op) {
    switch (op) {
    case 12: case 43: case 61: case 81: case 83: case 109: case 110: case 111: case 112: case 113: case 114: case 124: case 126: case 128: case 130: case 132: case 134: case 135:
    case 137: case 139: case 169: case 194: case 195: case 196: case 197: case 198: case 199: case 200: case 201: case 202: case 245:
        return 1;
    }
    return 0;
}

// Patch the module in *spv (*nw words) for the pool variables v[0..nv): returns a new malloc'd module (the caller frees the old) and its length in *nw, or NULL (spv_pool_err
// says why; the module is left as it is). cap: the texture arrays' length (a power of two is not required)
static uint32_t *spv_pool_patch(const uint32_t *spv, size_t *nw, spv_pool_var *v, int nv, uint32_t cap) {
    size_t n = *nw; spv_pool_err = NULL;
    if (n < 5 || spv[0] != 0x07230203u) { spv_pool_err = "not SPIR-V"; return NULL; }
    uint32_t bound = spv[3];
    // index the instructions
    size_t ni = 0, ci = 64; size_t *at = malloc(ci * sizeof *at);
    for (size_t i = 5; i < n;) { uint32_t c = spv[i] >> 16; if (!c || i + c > n) { free(at); spv_pool_err = "bad instruction length"; return NULL; }
        if (ni == ci) { ci *= 2; at = realloc(at, ci * sizeof *at); } at[ni++] = i; i += c; }
    #define OP(k) (spv[at[k]] & 0xffff)
    #define WC(k) (spv[at[k]] >> 16)
    #define W(k, j) (spv[at[k] + (j)])
    // where things are: the last capability / extension / ext-inst-import, the memory model, the last annotation; GLSL.std.450; a 32-bit unsigned int type
    size_t last_cap = 0, mm = 0, last_ann = 0; uint32_t glsl = 0, u32 = 0; int has_cap[3] = {0};
    for (size_t k = 0; k < ni; k++) {
        uint32_t op = OP(k);
        if (op == 17) { last_cap = k; uint32_t c = W(k, 1); has_cap[0] |= c == 5301; has_cap[1] |= c == 5307; has_cap[2] |= c == 29; }
        else if (op == 11) { if (WC(k) >= 3 && !strcmp((const char *)&W(k, 2), "GLSL.std.450")) glsl = W(k, 1); }
        else if (op == 14) mm = k;
        else if (op == 71 || op == 72) last_ann = k;
        else if (op == 21 && WC(k) == 4 && W(k, 2) == 32 && W(k, 3) == 0) u32 = W(k, 1);
    }
    int has_ext = 0;
    for (size_t k = 0; k < ni; k++) if (OP(k) == 10 && !strcmp((const char *)&W(k, 1), "SPV_EXT_descriptor_indexing")) has_ext = 1;
    if (!mm || !last_ann) { free(at); spv_pool_err = "no memory model or annotations"; return NULL; }
    // the variables: their Binding / DescriptorSet decorations and their OpVariable
    uint32_t var[16]; size_t dset[16], dbind[16], vdef[16]; if (nv > 16) { free(at); spv_pool_err = "more than 16 pool variables"; return NULL; }
    for (int p = 0; p < nv; p++) {
        var[p] = 0; dset[p] = dbind[p] = vdef[p] = (size_t)-1;
        for (size_t k = 0; k < ni; k++) if (OP(k) == 71 && WC(k) == 4 && W(k, 2) == 33 && W(k, 3) == v[p].binding) { var[p] = W(k, 1); dbind[p] = k; }
        if (!var[p]) { free(at); spv_pool_err = "no variable with the pool's binding"; return NULL; }
        for (size_t k = 0; k < ni; k++) {
            if (OP(k) == 71 && WC(k) == 4 && W(k, 1) == var[p] && W(k, 2) == 34) dset[p] = k;
            if (OP(k) == 59 && WC(k) >= 4 && W(k, 2) == var[p]) vdef[p] = k;
        }
        if (dset[p] == (size_t)-1 || vdef[p] == (size_t)-1) { free(at); spv_pool_err = "pool variable without a set or a definition"; return NULL; }
    }
    // the variables' types: pointer → array → image (class) or sampler
    uint32_t arr_elem[16], len_type[16];
    for (int p = 0; p < nv; p++) {
        uint32_t pt = W(vdef[p], 1), at_ = 0, el = 0, lc = 0; v[p].cls = -1;
        for (size_t k = 0; k < ni; k++) if (OP(k) == 32 && W(k, 1) == pt) at_ = W(k, 3);
        for (size_t k = 0; k < ni; k++) if (OP(k) == 28 && W(k, 1) == at_) { el = W(k, 2); lc = W(k, 3); }
        if (!el) { free(at); spv_pool_err = "pool variable is not a pointer to an array"; return NULL; }
        arr_elem[p] = el; len_type[p] = 0;
        for (size_t k = 0; k < ni; k++) if (OP(k) == 43 && W(k, 2) == lc) len_type[p] = W(k, 1);
        int found = 0;
        for (size_t k = 0; k < ni; k++) {
            if (OP(k) == 26 && W(k, 1) == el) { found = 1; if (!v[p].samp) { free(at); spv_pool_err = "a texture pool of samplers"; return NULL; } }
            if (OP(k) == 25 && W(k, 1) == el) {
                found = 1; if (v[p].samp) { free(at); spv_pool_err = "a sampler pool of images"; return NULL; }
                uint32_t st = W(k, 2), dim = W(k, 3), arrayed = W(k, 5), ms = W(k, 6), sampled = W(k, 7); int flt = 0;
                for (size_t q = 0; q < ni; q++) if (OP(q) == 22 && W(q, 1) == st) flt = 1;
                if (!flt || ms || sampled != 1) { free(at); spv_pool_err = "a texture pool of a class the pool set lacks (not float, multisampled or storage)"; return NULL; }
                v[p].cls = dim == 1 && !arrayed ? SPV_PC_2D : dim == 1 ? SPV_PC_2DA : dim == 2 && !arrayed ? SPV_PC_3D : dim == 3 && !arrayed ? SPV_PC_CUBE : -1;
                if (v[p].cls < 0) { free(at); spv_pool_err = "a texture pool of a class the pool set lacks (1D, cube array, ...)"; return NULL; }
            }
        }
        if (!found) { free(at); spv_pool_err = "pool element type not found"; return NULL; }
        if (!len_type[p]) len_type[p] = u32;
        if (!len_type[p]) { free(at); spv_pool_err = "no 32-bit integer type for the length"; return NULL; }
    }
    // a pool variable used other than through an access chain (loaded whole, stored, copied, passed to a function, a phi): not patched. Only the operands that are ids of
    // such instructions are looked at (a literal can equal the variable's id)
    for (size_t k = 0; k < ni; k++) {
        uint32_t op = OP(k), lo = 0, hi = 0;
        if (op == 61 || op == 83) lo = hi = 3; else if (op == 62 || op == 63) { lo = 1; hi = 2; } else if (op == 57) { lo = 4; hi = WC(k) - 1; } else if (op == 67) lo = hi = 3;
        else if (op == 245) { lo = 3; hi = WC(k) - 1; } else continue;
        for (uint32_t j = lo; j <= hi && j < WC(k); j++) for (int p = 0; p < nv; p++) if (W(k, j) == var[p]) { free(at); spv_pool_err = "a pool variable used other than through an access chain"; return NULL; }
    }
    // the inserts
    size_t nins = 0, cins = 64; spv_ins *ins = malloc(cins * sizeof *ins);
    #define INS(pos, ...) do { if (nins == cins) { cins *= 2; ins = realloc(ins, cins * sizeof *ins); } uint32_t w_[] = {__VA_ARGS__}; ins[nins].at = (pos); ins[nins].n = (int)(sizeof w_ / 4); memcpy(ins[nins].w, w_, sizeof w_); nins++; } while (0)
    uint32_t *rep = calloc(n, sizeof *rep); uint8_t *repd = calloc(n, 1);   // word replacements: rep[word] when repd[word]
    #define REP(word, val) do { rep[word] = (val); repd[word] = 1; } while (0)
    // header: capabilities, the extension, GLSL.std.450
    size_t after_cap = last_cap + 1;
    if (!has_cap[2]) INS(after_cap, (2u << 16) | 17, 29);
    if (!has_cap[0]) INS(after_cap, (2u << 16) | 17, 5301);
    if (!has_cap[1]) INS(after_cap, (2u << 16) | 17, 5307);
    if (!has_ext) { // "SPV_EXT_descriptor_indexing" = 27 characters + NUL = 7 words
        uint32_t s[7] = {0}; memcpy(s, "SPV_EXT_descriptor_indexing", 27);
        INS(after_cap, (8u << 16) | 10, s[0], s[1], s[2], s[3], s[4], s[5], s[6]); }
    if (!glsl) { glsl = bound++; uint32_t s[4] = {0}; memcpy(s, "GLSL.std.450", 12); INS(mm, (6u << 16) | 11, glsl, s[0], s[1], s[2], s[3]); }
    // per variable: set and binding, and for a texture pool a new length / array / pointer before its OpVariable
    for (int p = 0; p < nv; p++) {
        REP(at[dset[p]] + 3, SPV_POOL_SET);
        REP(at[dbind[p]] + 3, v[p].samp ? SPV_POOL_SAMP : (uint32_t)v[p].cls);
        if (v[p].samp) continue;
        uint32_t c = bound++, a = bound++, pt = bound++;
        INS(vdef[p], (4u << 16) | 43, len_type[p], c, cap);
        INS(vdef[p], (4u << 16) | 28, a, arr_elem[p], c);
        INS(vdef[p], (4u << 16) | 32, pt, 0 /* UniformConstant */, a);
        REP(at[vdef[p]] + 1, pt);
    }
    // per access chain: the clamp (texture pools) and NonUniform on the index, the chain, the loads of it and the sampled images made of those
    uint32_t nu[1024]; int nnu = 0;
    struct { uint32_t type, c; } cm[8]; int ncm = 0;   // cap - 1 as a constant of each index type
    size_t first_fn = 0; for (size_t k = 0; k < ni && !first_fn; k++) if (OP(k) == 54) first_fn = k;
    if (!first_fn) { free(at); free(ins); free(rep); free(repd); spv_pool_err = "no function"; return NULL; }
    for (size_t k = 0; k < ni; k++) {
        uint32_t op = OP(k); if ((op != 65 && op != 66) || WC(k) != 5) continue;
        int p = -1; for (int q = 0; q < nv; q++) if (W(k, 3) == var[q]) p = q;
        if (p < 0) continue;
        uint32_t idx = W(k, 4), chain = W(k, 2);
        if (!v[p].samp) {
            uint32_t it = 0;
            for (size_t q = 0; q < ni && !it; q++) if (spv_value_op(OP(q)) && WC(q) >= 3 && W(q, 2) == idx) it = W(q, 1);
            int isint = 0; for (size_t q = 0; q < ni; q++) if (OP(q) == 21 && W(q, 1) == it && W(q, 2) == 32) isint = 1;
            if (!isint) { free(at); free(ins); free(rep); free(repd); spv_pool_err = "a pool index whose type is not a 32-bit integer the patch can follow"; return NULL; }
            int m = -1; for (int q = 0; q < ncm; q++) if (cm[q].type == it) m = q;
            if (m < 0) { if (ncm == 8) { free(at); free(ins); free(rep); free(repd); spv_pool_err = "too many index types"; return NULL; }
                         m = ncm++; cm[m].type = it; cm[m].c = bound++; INS(first_fn, (4u << 16) | 43, it, cm[m].c, cap - 1); }
            uint32_t cl = bound++;
            INS(k, (7u << 16) | 12, it, cl, glsl, 38 /* UMin */, idx, cm[m].c);
            REP(at[k] + 4, cl); idx = cl;
        }
        if (nnu < 1020) { nu[nnu++] = idx; nu[nnu++] = chain; }
        for (size_t q = 0; q < ni; q++) if (OP(q) == 61 && WC(q) >= 4 && W(q, 3) == chain) {
            uint32_t ld = W(q, 2); if (nnu < 1022) nu[nnu++] = ld;
            for (size_t r = 0; r < ni; r++) if (OP(r) == 86 && (W(r, 3) == ld || W(r, 4) == ld) && nnu < 1023) nu[nnu++] = W(r, 2);
        }
    }
    // (the cap - 1 constants went before the first function: the end of the global section)
    for (int q = 0; q < nnu; q++) { int dup = 0; for (int r = 0; r < q; r++) dup |= nu[r] == nu[q]; if (!dup) INS(last_ann + 1, (3u << 16) | 71, nu[q], 5300 /* NonUniform */); }
    // write the module: instruction k's inserts first (in the order made), then k itself with its replaced words
    size_t extra = 0; for (size_t q = 0; q < nins; q++) extra += (size_t)ins[q].n;
    uint32_t *out = malloc((n + extra) * sizeof *out); size_t o = 0;
    memcpy(out, spv, 5 * sizeof *out); o = 5; out[3] = bound;
    for (size_t k = 0; k <= ni; k++) {
        for (size_t q = 0; q < nins; q++) if (ins[q].at == k) { memcpy(out + o, ins[q].w, (size_t)ins[q].n * 4); o += (size_t)ins[q].n; }
        if (k == ni) break;
        for (uint32_t j = 0; j < WC(k); j++) { size_t wi = at[k] + j; out[o++] = repd[wi] ? rep[wi] : spv[wi]; }
    }
    free(at); free(ins); free(rep); free(repd);
    *nw = o; return out;
    #undef OP
    #undef WC
    #undef W
    #undef INS
    #undef REP
}
#endif
