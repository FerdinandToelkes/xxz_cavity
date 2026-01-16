"""
This script generates a coverage report in LCOV format for the Dmrg module
This can be used by the VS Code Coverage Gutters extension to display coverage info

Execute this script by running:

    julia_itensors --project=. scripts/coverage.jl

within the methods/Dmrg directory or from the REPL with 

    julia> include("scripts/coverage.jl")

Use the command pallette "Coverage Gutters: Watch" to use the generated lcov.info file
to display coverage info directly in the editor
"""

using Pkg, Coverage, Coverage.LCOV

# run tests with coverage
Pkg.test(coverage=true)

# gather coverage data from the target_dir and write to file_name
target_dir = "src"
file_name = "lcov.info"
cov = process_folder(target_dir)
LCOV.writefile(file_name, cov)

# clean up .cov files
clean_folder(target_dir)
println("Coverage report generated at $(joinpath(pwd(), file_name))")