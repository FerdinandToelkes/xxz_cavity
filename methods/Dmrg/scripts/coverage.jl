# This script generates a coverage report in LCOV format for the Dmrg module
# This can be used by the VS Code Coverage Gutters extension to display coverage info

# Use the command pallette "Coverage Gutters: Watch" to use the generated coverage.info file
# to display coverage info directly in the editor

# Execute this script by running:
# julia_itensors --project=. scripts/coverage.jl
# within the methods/Dmrg directory or from the REPL with 
# include("scripts/coverage.jl")


using Coverage, Coverage.LCOV

cov = process_folder("src")
LCOV.writefile("lcov.info", cov)
