/*
 * FBtCNN filter-bank preprocessing via SOS filtfilt.
 *
 * Replicates the host-side _filter_bank() from src/models/fbtcnn.py:
 *   - 4 bands of 6th-order Butterworth bandpass (scipy.signal.butter)
 *   - Zero-phase filtering (scipy.signal.filtfilt with odd extension)
 *
 * Uses the existing sosfilt_mcu (common component) for the actual
 * filtfilt implementation, which is already verified against scipy
 * for the FBCCA classical backend.
 *
 * Contract (frozen from host training code):
 *   1. Input:  [T=512, C=8] raw EEG, float32, TC row-major
 *   2. For each of 4 bands, for each of 8 channels:
 *      - Extract channel time series
 *      - Apply SOS filtfilt with odd extension (edge=18)
 *      - Store filtered result
 *   3. Output: NHWC [1, C=8, T=512, B=4] row-major
 *      - output[c * 512 * 4 + t * 4 + b] = filtered sample
 *   4. Total: 8 × 512 × 4 = 16384 floats
 */
#include "fbtcnn_preprocess.h"
#include "fbtcnn_filter_generated.h"
#include "sosfilt_mcu.h"

#define LEN_EXT (FBTCNN_INPUT_T + 2 * FBTCNN_EDGE)  /* 512 + 78 = 590 */

void fbtcnn_preprocess(const float *input_tc, float *output)
{
    static double ch_buf[FBTCNN_INPUT_T];
    static double ext_buf[LEN_EXT];

    for (int band = 0; band < FBTCNN_NUM_BANDS; band++) {
        for (int c = 0; c < FBTCNN_INPUT_C; c++) {

            for (int t = 0; t < FBTCNN_INPUT_T; t++)
                ch_buf[t] = (double)input_tc[t * FBTCNN_INPUT_C + c];

            sosfilt_odd_ext_1d(ch_buf, FBTCNN_INPUT_T, FBTCNN_EDGE, ext_buf);
            sosfilt_filtfilt(FBTCNN_SOS[band], FBTCNN_ZI[band],
                             FBTCNN_N_SOS, ext_buf, LEN_EXT);

            /*
             * Store into NHWC layout: output[c][t][band]
             * = output[c * INPUT_T * NUM_BANDS + t * NUM_BANDS + band]
             */
            float *dst = output + c * FBTCNN_INPUT_T * FBTCNN_NUM_BANDS;
            for (int t = 0; t < FBTCNN_INPUT_T; t++)
                dst[t * FBTCNN_NUM_BANDS + band] = (float)ext_buf[FBTCNN_EDGE + t];
        }
    }
}
