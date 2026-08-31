# Vendored front-end libraries

The mission console loads its two runtime libraries from here rather than from a
CDN. That is a deliberate choice for a safety-adjacent display:

* **It works offline.** An operations console that goes blank because a CDN is
  unreachable is worse than useless — it fails at exactly the moment the network
  is degraded.
* **It allows a strict Content-Security-Policy.** With no third-party script
  origin, the console can run under `script-src 'self'`, which removes an entire
  class of supply-chain and injection risk.
* **The bytes cannot change under you.** A CDN version can be re-published; a
  vendored file is whatever is in this commit.

| File               | Library      | Version | Upstream                                                              |
|--------------------|--------------|---------|-----------------------------------------------------------------------|
| `three.min.js`     | three.js     | r128    | https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js      |
| `satellite.min.js` | satellite.js | 5.0.0   | https://cdnjs.cloudflare.com/ajax/libs/satellite.js/5.0.0/satellite.min.js |

Integrity of the files as vendored (SHA-384, the value you would put in an
`integrity=` attribute if you ever go back to a CDN):

```
three.min.js      sha384-CI3ELBVUz9XQO+97x6nwMDPosPR5XvsxW2ua7N1Xeygeh1IxtgqtCkGfQY9WWdHu
satellite.min.js  sha384-e6V1JVpc+kVNTxT0gEOla7yL4c92Romsw7itPoipfzHbEjzUwPhTv6oeVDMHHzqT
```

## Updating

Download the new version, verify its published integrity hash, replace the file,
update the table above, and re-check the console renders. Do not edit these
files by hand.
