# Pluto behind Jupyter's server proxy: /pluto/ on Binder.
c = get_config()  # noqa: F821
c.ServerProxy.servers = {
    "pluto": {
        "command": [
            "julia", "--project=/home/jovyan/notebook", "-e",
            "import Pluto; Pluto.run(host=\"127.0.0.1\", port={port}, "
            "launch_browser=false, require_secret_for_open_links=false, "
            "require_secret_for_access=false, dismiss_update_notification=true)",
        ],
        "timeout": 120,
        "absolute_url": False,
        "launcher_entry": {"title": "Pluto.jl"},
    }
}
