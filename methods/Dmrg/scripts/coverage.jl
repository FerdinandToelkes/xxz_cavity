"""
This script generates a coverage report in LCOV format for the Dmrg module
This can be used by the VS Code Coverage Gutters extension to display coverage info

Execute this script by running

    julia --sysimage ~/.julia/sysimages/sys_itensors.so --project=. scripts/coverage.jl

from the terminal or from the REPL by running:

    julia> include("scripts/coverage.jl")

within the methods/Dmrg directory.

Use the command pallette "Coverage Gutters: Watch" to use the generated lcov.info file
to display coverage info directly in the editor
"""

using Pkg
using Coverage
using Coverage.LCOV

target_dirs = ["src", "test"]
file_name = "lcov.info"

function run_coverage()
    try
        # run tests with coverage enabled
        Pkg.test(coverage = true)

        # gather coverage data and write to file
        cov = Vector{Coverage.FileCoverage}()
        for dir in target_dirs
            append!(cov, process_folder(dir))
        end
        LCOV.writefile(file_name, cov)

        println("Coverage report generated at $(joinpath(pwd(), file_name))")

    catch err
        # make failure visible but do not suppress it
        @error "Test or coverage generation failed" exception = err
        rethrow()

    finally
        # always clean up .cov files
        for dir in target_dirs
            try
                clean_folder(dir)
            catch cleanup_err
                @warn "Failed to clean coverage files in $dir" exception = cleanup_err
            end
        end
    end
end

run_coverage()
nothing
