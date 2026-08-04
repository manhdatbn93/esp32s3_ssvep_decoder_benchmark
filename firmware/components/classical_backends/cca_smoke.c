/*
 * CCA smoke backend — wires cca_forward into the unified smoke interface
 * and exports all flash-resident smoke vectors + expected outputs.
 */
#include "cca_forward.h"
#include "smoke_vectors_cca.h"

int cca_smoke_init(void) {
    return cca_forward_init();
}

int cca_smoke_run(const float *input_tc, float *scores_out) {
    return cca_forward_run(input_tc, scores_out);
}

void cca_smoke_cleanup(void) {
    cca_forward_cleanup();
}

const int cca_smoke_n_vectors = SMOKE_CCA_N_VECTORS;

const float *const cca_smoke_vectors[] = {
    SMOKE_CCA_VEC0, SMOKE_CCA_VEC1, SMOKE_CCA_VEC2,
    SMOKE_CCA_VEC3, SMOKE_CCA_VEC4
};

const int cca_smoke_expected_argmax[5] = { 1, 9, 0, 0, 3 };

const int   cca_smoke_has_expected_scores = SMOKE_CCA_HAS_EXPECTED_SCORES;
const float *cca_smoke_expected_scores_flat = &SMOKE_CCA_EXPECTED_SCORES[0][0];
