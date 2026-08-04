/*
 * TRCA smoke backend — wires trca_forward into the unified smoke interface
 * and exports all flash-resident smoke vectors + expected outputs.
 */
#include "trca_forward.h"
#include "smoke_vectors_trca.h"

int trca_smoke_init(void) {
    return trca_forward_init();
}

int trca_smoke_run(const float *input_tc, float *scores_out) {
    return trca_forward_run(input_tc, scores_out);
}

void trca_smoke_cleanup(void) {
    trca_forward_cleanup();
}

const int trca_smoke_n_vectors = SMOKE_TRCA_N_VECTORS;

const float *const trca_smoke_vectors[] = {
    SMOKE_TRCA_VEC0, SMOKE_TRCA_VEC1, SMOKE_TRCA_VEC2,
    SMOKE_TRCA_VEC3, SMOKE_TRCA_VEC4
};

const int *trca_smoke_expected_argmax = SMOKE_TRCA_EXPECTED_ARGMAX;

const int trca_smoke_has_expected_scores = SMOKE_TRCA_HAS_EXPECTED_SCORES;
#if SMOKE_TRCA_HAS_EXPECTED_SCORES
const float *trca_smoke_expected_scores_flat = &SMOKE_TRCA_EXPECTED_SCORES[0][0];
#else
const float *trca_smoke_expected_scores_flat = (const float *)0;
#endif
