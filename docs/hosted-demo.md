# Hosted demo (static)

The GitHub Pages site is fully static. There is no server or secrets. As a result:

- “Suggest with AI (local)” is disabled and shows a tip.
- Use “Open prompt in ChatGPT” to copy a structured prompt and paste into ChatGPT.
- Or run the local API to enable AI suggestions while browsing the hosted site.

Deployment uses `pages.yml` with `actions/configure-pages`, `actions/upload-pages-artifact`, and `actions/deploy-pages`.

