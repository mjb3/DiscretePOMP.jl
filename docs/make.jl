using Documenter, DiscretePOMP

makedocs(sitename="DiscretePOMP.jl docs", pages = ["index.md", "models.md", "examples.md", "manual.md"])

## nb. called by GitHub Actions wf
# - local version deploys to build dir
deploydocs(
    repo = "github.com/mjb3/DiscretePOMP.jl.git",
)
