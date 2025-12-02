import pkg_resources

with open("requirements.txt", "w") as f:
    for dist in pkg_resources.working_set:
        f.write(dist.project_name + "\n")
