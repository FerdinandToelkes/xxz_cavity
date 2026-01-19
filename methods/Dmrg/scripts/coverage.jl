"""
This script generates a coverage report in LCOV format for the Dmrg module
This can be used by the VS Code Coverage Gutters extension to display coverage info

Execute this script by running from the REPL by running: 

    julia> include("scripts/coverage.jl")

within the methods/Dmrg directory.

Use the command pallette "Coverage Gutters: Watch" to use the generated lcov.info file
to display coverage info directly in the editor
"""

using Pkg, Coverage, Coverage.LCOV

# run tests with coverage
Pkg.test(coverage=true)

# gather coverage data from the target_dir and write to file_name
target_dirs = ["src", "test"]
file_name = "lcov.info"
cov = Vector{Coverage.FileCoverage}()
for dir in target_dirs
    tmp = process_folder(dir)
    append!(cov, tmp)
end
LCOV.writefile(file_name, cov)

# clean up .cov files
for dir in target_dirs
    clean_folder(dir)
end
println("Coverage report generated at $(joinpath(pwd(), file_name))")