---
name: env-headless-render-tools
description: how to screenshot an HTML page and run Node on this box — firefox is a snap that can't read /tmp; node isn't installed but a pip wheel provides it
metadata:
  type: reference
---
- **No `node` on PATH.** Get a binary without touching the project env:
  `uv run --with nodejs-wheel-binaries python -c "import nodejs_wheel,os;print(os.path.dirname(nodejs_wheel.__file__))"`
  then use `<that dir>/bin/node` (v24). `uvx --from nodejs-wheel-binaries node` does NOT work (no entry point).
- **Headless screenshots:** `/usr/bin/firefox` is a snap, so it cannot read `/tmp` or arbitrary
  profile dirs ("Could not find profile folder"). Copy the page into `~/snap/firefox/common/<dir>/`
  and run `firefox --headless --no-remote --profile <dir>/prof --window-size=1280,11000
  --screenshot <dir>/shot.png file://<dir>/page.html`, then delete the dir.
