# Review Guard Deployment (Render + Custom Domain)

This project is preconfigured for Render with [`render.yaml`](./render.yaml).

## 1. Push this project to GitHub

Render deploys from a Git repository. Push this folder to a GitHub repo first.

## 2. Create the web service in Render

1. In Render, click `New` -> `Blueprint`.
2. Select your GitHub repo.
3. Render will detect `render.yaml` and create service `reviewguard`.
4. Wait for deploy to complete.
5. Verify:
   - `https://<your-render-url>/health` returns `{"status":"ok"}`
   - `https://<your-render-url>/` loads the Review Guard console

## 3. Buy/register your domain

Buy a domain from any registrar (Cloudflare, Namecheap, GoDaddy, etc.).
Examples: `reviewguard.ai`, `reviewguard.co.ke`.

## 4. Attach domain in Render

1. Open service `reviewguard` in Render.
2. Go to `Settings` -> `Custom Domains`.
3. Add:
   - `reviewguard.yourdomain.com` (subdomain, recommended first)
   - Optionally add apex domain `yourdomain.com`
4. Render shows required DNS records (CNAME/A/ALIAS depending on registrar).

## 5. Add DNS records at registrar

Create the exact records Render gives you, then wait for DNS propagation.

## 6. HTTPS

Render provisions TLS certificate automatically after domain verification.

## Notes

- If startup fails due to memory limits on free tiers, upgrade the instance type.
- Default model paths are set via env vars in `render.yaml`.
- `playwright` is optional in this app; the API works without it.
