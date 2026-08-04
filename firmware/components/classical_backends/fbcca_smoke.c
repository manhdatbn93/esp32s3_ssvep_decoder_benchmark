/*
 * FBCCA smoke backend — wires fbcca_forward into the unified smoke interface
 * and exports all flash-resident smoke vectors + expected outputs.
 */
#include "fbcca_forward.h"
#include "smoke_vectors_fbcca.h"

int fbcca_smoke_init(void) {
    return fbcca_forward_init();
}

int fbcca_smoke_run(const float *input_tc, float *scores_out) {
    return fbcca_forward_run(input_tc, scores_out);
}

void fbcca_smoke_cleanup(void) {
    fbcca_forward_cleanup();
}

const int fbcca_smoke_n_vectors = SMOKE_FBCCA_N_VECTORS;

const float *const fbcca_smoke_vectors[] = {
    SMOKE_FBCCA_VEC0, SMOKE_FBCCA_VEC1, SMOKE_FBCCA_VEC2,
    SMOKE_FBCCA_VEC3, SMOKE_FBCCA_VEC4
};

const int fbcca_smoke_expected_argmax[5] = { 5, 8, 0, 0, 5 };

const int   fbcca_smoke_has_expected_scores = SMOKE_FBCCA_HAS_EXPECTED_SCORES;
const float *fbcca_smoke_expected_scores_flat = &SMOKE_FBCCA_EXPECTED_SCORES[0][0];
