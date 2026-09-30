# DataPulse static dashboard

`index.html` loads the CSV copies in `data/` and uses Chart.js from a CDN. Run it through a web server; opening the file directly blocks browser CSV requests.

## Run locally

Open the `dashboard` folder in VS Code and use the Live Server extension, or run this from that folder with Node.js installed:

```powershell
npx serve .
```

Open the local URL printed by the command.

## Publish with GitHub Pages

1. Commit and push the complete `dashboard` folder, including `dashboard/data/`.
2. In the GitHub repository, open **Settings** > **Pages**.
3. Select **Deploy from a branch**, choose your default branch, and choose the repository root.
4. Open `https://YOUR-USERNAME.github.io/YOUR-REPOSITORY/dashboard/` after deployment.

The dashboard uses relative data paths, so the copied CSV files load on GitHub Pages without changes.
