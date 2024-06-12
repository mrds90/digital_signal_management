/**
 * @file filter_math.h
 * @author Marcos Dominguez
 *
 * @brief Module description
 *
 * @version X.Y
 * @date 2024-06-12
 */

#ifndef FILTER_MATH_H
#define FILTER_MATH_H

/*========= [DEPENDENCIES] =====================================================*/

#include <stdint.h>

/*========= [PUBLIC MACRO AND CONSTANTS] =======================================*/

/*========= [PUBLIC DATA TYPE] =================================================*/

/*========= [PUBLIC FUNCTION DECLARATIONS] =====================================*/

/**
 * @brief Multiplies two vectors element-wise in Q15 format.
 *
 * This function performs an element-wise multiplication of two input vectors (op1 and op2),
 * both of which are assumed to be in Q15 format (fixed-point with 15 fractional bits).
 * The result of each multiplication is right-shifted by 15 bits to maintain the Q15 format.
 * 
 * @param op1 Pointer to the first input vector.
 * @param op2 Pointer to the second input vector.
 * @param result Pointer to the output vector where results are stored.
 * @param size The number of elements in each input vector.
 */
void FILTER_MATH_MultQ15(const int32_t *op1, const int32_t *op2, int32_t *result, uint16_t size);

/**
 * @brief Multiplies two vectors element-wise in Q15 format and sums the results.
 *
 * This function first multiplies two input vectors (op1 and op2) element-wise using the
 * FILTER_MATH_MultQ15 function. It then sums the resulting elements and returns the total sum.
 * An initial value can be specified, which is added to the sum of the products.
 * 
 * @param op1 Pointer to the first input vector.
 * @param op2 Pointer to the second input vector.
 * @param result Pointer to the output vector where intermediate results are stored.
 * @param size The number of elements in each input vector.
 * @param init_value The initial value to be added to the sum of the products.
 * @return The sum of the element-wise products, plus the initial value.
 */
int32_t FILTER_MATH_MultSumQ15(const int32_t *op1, const int32_t *op2, int32_t *result, uint16_t size, int32_t init_value);

#endif  /* FILTER_MATH_H */
