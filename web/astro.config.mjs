import { defineConfig } from 'astro/config';
import tailwind from "@astrojs/tailwind";

export default defineConfig({
  site: 'https://florin1975.github.io',
  base: '/aibasedria',
  integrations: [tailwind({ applyBaseStyles: true })],
});
