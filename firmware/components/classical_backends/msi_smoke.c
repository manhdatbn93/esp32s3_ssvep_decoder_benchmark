/*
 * MSI smoke backend — wires msi_forward into the unified smoke interface
 * and exports all flash-resident smoke vectors + expected outputs.
 */
#include "msi_forward.h"
#include "smoke_vectors_msi.h"

int msi_smoke_init(void) {
    return msi_forward_init();
}

int msi_smoke_run(const float *input_tc, float *scores_out) {
    return msi_forward_run(input_tc, scores_out);
}

void msi_smoke_cleanup(void) {
    msi_forward_cleanup();
}

const int msi_smoke_n_vectors = SMOKE_MSI_N_VECTORS;

const float *const msi_smoke_vectors[] = {
    SMOKE_MSI_VEC0, SMOKE_MSI_VEC1, SMOKE_MSI_VEC2,
    SMOKE_MSI_VEC3, SMOKE_MSI_VEC4
};

const int msi_smoke_expected_argmax[5] = { 0, 3, 0, 0, 0 };

const int   msi_smoke_has_expected_scores = SMOKE_MSI_HAS_EXPECTED_SCORES;
const float *msi_smoke_expected_scores_flat = &SMOKE_MSI_EXPECTED_SCORES[0][0];
