# Vendored libraries

Copied unchanged from the official npm release tarballs. `npm pack` verified each tarball
against the registry's `dist.integrity` (SHA-512) before extraction.

| Library | Version | Source tarball | Tarball integrity (npm) | Files kept | Licence |
|---|---|---|---|---|---|
| Leaflet | 1.9.4 | https://registry.npmjs.org/leaflet/-/leaflet-1.9.4.tgz | `sha512-nxS1ynzJOmOlHp+iL3FyWqK89GtNL8U8rvlMOsQdTTssxZwCXh8N2NB3GDQOL+YR3XnWyZAxwQixURb+FA74PA==` | `dist/leaflet.js`, `dist/leaflet.css`, `dist/images/*`, `LICENSE` | BSD-2-Clause (`leaflet-1.9.4/LICENSE`) |
| Chart.js | 4.6.2 | https://registry.npmjs.org/chart.js/-/chart.js-4.6.2.tgz | `sha512-xachB+pPHjp0syJi7llUBA1w3kMYl6NF3o7eVrvjnWCT+IB2zMaF2pX5UYDiwbbfsioFEIRnAjreNOMsGj5TIA==` | `dist/chart.umd.min.js`, `LICENSE.md` | MIT (`chart.js-4.6.2/LICENSE.md`) |

SHA-256 of the vendored files:

```
db49d009c841f5ca34a888c96511ae936fd9f5533e90d8b2c4d57596f4e5641a  leaflet-1.9.4/leaflet.js
a7837102824184820dfa198d1ebcd109ff6d0ff9a2672a074b9a1b4d147d04c6  leaflet-1.9.4/leaflet.css
53e8dc25862014e4324741ca18fbe3611e11d42ef69f59f86ea8c5389647d4cb  leaflet-1.9.4/LICENSE
3f288f4f26882c26b58931fb586a9b4f30370b5de38d69dc005b0bd05b08e113  chart.js-4.6.2/chart.umd.min.js
41a84aa2caba645f966a18d9c2056b73e6d3a81d80bc0046bc0011a2634d4cce  chart.js-4.6.2/LICENSE.md
```

To upgrade: `npm pack <name>@<version>`, copy the same files into a new versioned folder,
update the paths in the HTML pages, and update this table.
