/**
 * @file filter_math.c
 * @author Marcos Dominguez
 *
 * @brief This module provides operations to sum and multiply vectors of fixed-point Q15 format in C.
 *
 * It includes functions for element-wise multiplication of two vectors and for summing the results
 * of such multiplications. These operations are useful in digital signal processing (DSP) applications
 * where fixed-point arithmetic is often used for performance reasons.
 *
 * @version X.Y
 * @date 2024-06-12
 */

/*========= [DEPENDENCIES] =====================================================*/

#include "filter_math.h"
#include <string.h>


/*========= [PRIVATE MACROS AND CONSTANTS] =====================================*/

/*========= [PRIVATE DATA TYPES] ===============================================*/

/*========= [TASK DECLARATIONS] ================================================*/

/*========= [PRIVATE FUNCTION DECLARATIONS] ====================================*/

/*========= [INTERRUPT FUNCTION DECLARATIONS] ==================================*/

/*========= [LOCAL VARIABLES] ==================================================*/

/*========= [STATE FUNCTION POINTERS] ==========================================*/

/*========= [PUBLIC FUNCTION IMPLEMENTATION] ===================================*/

void FILTER_MATH_MultQ15(const int32_t *op1, const int32_t *op2, int32_t *result, uint16_t size) {
    for (uint16_t i = 0; i < size; i++) {
        result[i] = (op1[i] * op2[i]) >> 15;
    }
}

int32_t FILTER_MATH_MultSumQ15(const int32_t *op1, const int32_t *op2, int32_t *result, uint16_t size, int32_t init_value) {
    FILTER_MATH_MultQ15(op1, op2, result, size);
    
    int32_t output = init_value;

    for (uint16_t i = 0; i < size; i++) {
        output += result[i];
    }

    return output;
}

/*========= [PRIVATE FUNCTION IMPLEMENTATION] ==================================*/

/*========= [INTERRUPT FUNCTION IMPLEMENTATION] ================================*/
