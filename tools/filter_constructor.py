import argparse
from datetime import datetime
import os
from typing import List, Tuple

class FilterGenerator:
    # Class-level constants for templates
    c_template: str = """/**
 * @file {source_file}.c
 * @author Marcos Dominguez
 *
 * @brief Module description
 *
 * @version X.Y
 * @date {date}
 */

/*========= [DEPENDENCIES] =====================================================*/

#include "{header_file}.h"
#include <string.h>
{additional_includes}

/*========= [PRIVATE MACROS AND CONSTANTS] =====================================*/

{macro_definitions}

#define F_TO_Q15(x)  (int32_t)((x) * (1 << 15))

#define NUM_SIZE {num_size}
#define DEN_SIZE {den_size}

/* Numerator coefficients in Q15 */
{num_defines}

/* Denominator coefficients in Q15 */
{den_defines}

/*========= [PRIVATE DATA TYPES] ===============================================*/

/*========= [TASK DECLARATIONS] ================================================*/

/*========= [PRIVATE FUNCTION DECLARATIONS] ====================================*/

/*========= [INTERRUPT FUNCTION DECLARATIONS] ==================================*/

/*========= [LOCAL VARIABLES] ==================================================*/

{input_buffer}
{output_buffer}

/*========= [STATE FUNCTION POINTERS] ==========================================*/

/*========= [PUBLIC FUNCTION IMPLEMENTATION] ===================================*/

{reset_function}

{function_definitions}

/*========= [PRIVATE FUNCTION IMPLEMENTATION] ==================================*/

/*========= [INTERRUPT FUNCTION IMPLEMENTATION] ================================*/
"""

    h_template: str = """/**
 * @file {header_file}.h
 * @author Marcos Dominguez
 *
 * @brief Module description
 *
 * @version X.Y
 * @date {date}
 */

#ifndef {header_guard}
#define {header_guard}

/*========= [DEPENDENCIES] =====================================================*/

#include <stdint.h>

/*========= [PUBLIC MACRO AND CONSTANTS] =======================================*/

/*========= [PUBLIC DATA TYPE] =================================================*/

/*========= [PUBLIC FUNCTION DECLARATIONS] =====================================*/

/**
 * @brief Resets the filter state.
 *
 * This function resets the internal state of the filter buffers.
 */
int32_t {file_upper}_Reset(void);

/**
 * @brief Filters input signal using a digital filter.
 *
 * This function implements a digital filter to process the input signal. It takes
 * the input signal in Q15 format and applies a numerator-denominator filter with
 * coefficients provided as Q15 fixed-point values.
 *
 * @param input Input signal in Q15 format.
 * @return Filtered output signal in Q15 format.
 */
int32_t {file_upper}_Filter(int32_t input);

#endif  /* {header_guard} */
"""

    def __init__(self, base_name: str, num_coeffs: List[float], den_coeffs: List[float], additional_includes: List[str] = None,
                 mul_elements: str = None, mul_elements_q15: str = None, mul_sum_elements_q15: str = None, mul_sub_elements_q15: str = None):
        """
        Initializes the FilterGenerator with base name, numerator and denominator coefficients,
        and optional additional includes and macro definitions.

        :param base_name: Base name for the generated files
        :param num_coeffs: List of numerator coefficients
        :param den_coeffs: List of denominator coefficients
        :param additional_includes: List of additional include files
        :param mul_elements: Custom definition for MUL_ELEMENTS macro
        :param mul_elements_q15: Custom definition for MUL_ELEMENTS_Q15 macro
        :param mul_sum_elements_q15: Custom definition for MUL_SUM_ELEMENTS_Q15 macro
        :param mul_sub_elements_q15: Custom definition for MUL_SUB_ELEMENTS_Q15 macro
        """
        self.base_name = base_name
        self.num_coeffs = num_coeffs
        self.den_coeffs = den_coeffs
        self.header_file = base_name
        self.source_file = base_name
        self.file_upper = base_name.upper()
        self.header_guard = f"{self.file_upper}_H"
        self.date = datetime.now().strftime("%Y-%m-%d")
        
        self.additional_includes = additional_includes if additional_includes else []
        
        # Macro definitions with default values
        self.mul_elements = mul_elements if mul_elements else "#define MUL_ELEMENTS(x, y)  ((x) * (y))"
        self.mul_elements_q15 = mul_elements_q15 if mul_elements_q15 else "#define MUL_ELEMENTS_Q15(x, y)  (MUL_ELEMENTS(x, y) >> 15)"
        self.mul_sum_elements_q15 = mul_sum_elements_q15 if mul_sum_elements_q15 else "#define MUL_SUM_ELEMENTS_Q15(x, y, z) ((z) + MUL_ELEMENTS_Q15((x), (y)))"
        self.mul_sub_elements_q15 = mul_sub_elements_q15 if mul_sub_elements_q15 else "#define MUL_SUB_ELEMENTS_Q15(x, y, z) ((z) - MUL_ELEMENTS_Q15((x), (y)))"
        
        self.num_size, self.den_size, self.num_defines, self.den_defines, self.function_definitions = self.generate_function_definitions()
        self.reset_function = self.generate_reset_function()
        self.input_buffer, self.output_buffer = self.generate_buffers()

    def add_include(self, include: str) -> None:
        """
        Adds an additional include file.

        :param include: The include file to add
        """
        self.additional_includes.append(include)
    
    def set_mul_elements(self, definition: str) -> None:
        """
        Sets a custom definition for the MUL_ELEMENTS macro.

        :param definition: Custom definition for MUL_ELEMENTS
        """
        self.mul_elements = f"#define MUL_ELEMENTS(x, y)  {definition}"

    def set_mul_elements_q15(self, definition: str) -> None:
        """
        Sets a custom definition for the MUL_ELEMENTS_Q15 macro.

        :param definition: Custom definition for MUL_ELEMENTS_Q15
        """
        self.mul_elements_q15 = f"#define MUL_ELEMENTS_Q15(x, y)  {definition}"

    def set_mul_sum_elements_q15(self, definition: str) -> None:
        """
        Sets a custom definition for the MUL_SUM_ELEMENTS_Q15 macro.

        :param definition: Custom definition for MUL_SUM_ELEMENTS_Q15
        """
        self.mul_sum_elements_q15 = f"#define MUL_SUM_ELEMENTS_Q15(x, y, z)  {definition}"

    def set_mul_sub_elements_q15(self, definition: str) -> None:
        """
        Sets a custom definition for the MUL_SUB_ELEMENTS_Q15 macro.

        :param definition: Custom definition for MUL_SUB_ELEMENTS_Q15
        """
        self.mul_sub_elements_q15 = f"#define MUL_SUB_ELEMENTS_Q15(x, y, z)  {definition}"

    def generate_c_file(self) -> str:
        """
        Generates the content for the .c file.

        :return: Generated C file content
        """
        additional_includes_str = "\n".join([f"#include {include}" for include in self.additional_includes])
        
        macro_definitions = "\n".join([
            f"#ifndef MUL_ELEMENTS\n{self.mul_elements}\n#endif",
            f"#ifndef MUL_ELEMENTS_Q15\n{self.mul_elements_q15}\n#endif",
            f"#ifndef MUL_SUM_ELEMENTS_Q15\n{self.mul_sum_elements_q15}\n#endif",
            f"#ifndef MUL_SUB_ELEMENTS_Q15\n{self.mul_sub_elements_q15}\n#endif"
        ])
        
        return self.c_template.format(
            source_file=self.source_file,
            num_defines=self.num_defines,
            den_defines=self.den_defines,
            num_size=self.num_size,
            den_size=self.den_size,
            function_definitions=self.function_definitions,
            date=self.date,
            header_file=self.header_file,
            file_upper=self.file_upper,
            reset_function=self.reset_function,
            additional_includes=additional_includes_str,
            macro_definitions=macro_definitions,
            input_buffer=self.input_buffer,
            output_buffer=self.output_buffer
        )

    def generate_h_file(self) -> str:
        """
        Generates the content for the .h file.

        :return: Generated header file content
        """
        return self.h_template.format(
            header_file=self.header_file,
            header_guard=self.header_guard,
            date=self.date,
            file_upper=self.file_upper
        )

    def generate_reset_function(self) -> str:
        """
        Generates the reset function definition.

        :return: Generated reset function definition
        """
        reset_function = f"""int32_t {self.file_upper}_Reset() {{
    memset(input_buffer, 0, sizeof(input_buffer));"""
        if self.den_size > 1:
                    reset_function +="""
    memset(output_buffer, 0, sizeof(output_buffer));
"""
    
        reset_function +="""}"""
        return reset_function

    def generate_buffers(self) -> Tuple[str, str]:
        """
        Generates the input and output buffer definitions based on the sizes of num_coeffs and den_coeffs.

        :return: Tuple containing input buffer and output buffer definitions
        """
        if self.num_size > 1:
            input_buffer = f"static int32_t input_buffer[NUM_SIZE] = {{[0 ... (NUM_SIZE - 1)] = 0}};"
        else:
            input_buffer = "static int32_t input_buffer[1] = {0};"

        if self.den_size > 1:
            output_buffer = f"static int32_t output_buffer[DEN_SIZE - 1] = {{[0 ... (DEN_SIZE - 2)] = 0}};"
        else:
            output_buffer = ""

        return input_buffer, output_buffer

    def generate_function_definitions(self) -> Tuple[int, int, str, str, str]:
        """
        Generates the filter function definitions and related information.

        :return: Tuple containing number of numerator coefficients, number of denominator coefficients,
                 numerator definitions, denominator definitions, and function code
        """
        num_size = len(self.num_coeffs)
        den_size = len(self.den_coeffs)
        
        num_defines = "\n".join([f"#define NUM{i} F_TO_Q15({self.num_coeffs[i]})" for i in range(num_size)])
        den_defines = "\n".join([f"#define DEN{i} F_TO_Q15({self.den_coeffs[i]})" for i in range(den_size)])  # DEN0 is included now
        
        function_code = f"""int32_t {self.file_upper}_Filter(int32_t input) {{
    /* Shift values in the input buffer */
"""
        if num_size > 1:
            function_code += """    for (int i = NUM_SIZE - 1; i > 0; --i) {
        input_buffer[i] = input_buffer[i - 1];
    }
"""
        function_code += f"""    input_buffer[0] = input;

    /* Calculate the numerator part */
    int32_t output = 0;
""" + "".join([f"    output = MUL_SUM_ELEMENTS_Q15(NUM{i}, input_buffer[{i}], output);\n" for i in range(num_size)]) 
        if den_size > 1:
            function_code +="""
    /* Calculate the denominator part */""" 
        function_code += f"""
""" + "".join([f"    output = MUL_SUB_ELEMENTS_Q15(DEN{i}, output_buffer[{i-1}], output);\n" for i in range(1, den_size)])

        if den_size > 1:
            function_code += f"""
    /* Shift values in the output buffer */
    for (int i = DEN_SIZE - 2; i > 0; --i) {{
        output_buffer[i] = output_buffer[i - 1];
    }}
    output_buffer[0] = output;
"""
        function_code += f"""
    return output;
}}"""
        return num_size, den_size, num_defines, den_defines, function_code

    def write_files(self, path_c: str, path_h: str) -> None:
        """
        Writes the generated C and header files to the specified paths.

        :param path_c: Path to the directory for the C file
        :param path_h: Path to the directory for the header file
        """
        c_file_path = os.path.join(path_c, f"{self.base_name}.c")
        h_file_path = os.path.join(path_h, f"{self.base_name}.h")
        
        c_content = self.generate_c_file()
        h_content = self.generate_h_file()
        
        # Ensure the directory exists
        os.makedirs(path_c, exist_ok=True)
        os.makedirs(path_h, exist_ok=True)
        
        with open(c_file_path, "w") as c_file:
            c_file.write(c_content)
            
        with open(h_file_path, "w") as h_file:
            h_file.write(h_content)
            
        print(f"Files {c_file_path} and {h_file_path} generated successfully.")


def main():
    parser = argparse.ArgumentParser(description="Generate .c and .h files based on templates.")
    parser.add_argument("-file", default="real_world_filter", help="Base name for the .c and .h files")
    parser.add_argument("-num_coeffs", nargs='+', type=float, default=[0.04976845243756167, 0.035050642374672925], help="Numerator coefficients")
    parser.add_argument("-den_coeffs", nargs='+', type=float, default=[1.0, -1.2631799459800208, 0.34799904079225535], help="Denominator coefficients")

    args = parser.parse_args()
    
    generator = FilterGenerator(args.file, args.num_coeffs, args.den_coeffs)
    generator.write_files("./", "./")  # Specify the directories for .c and .h files

if __name__ == "__main__":
    main()
