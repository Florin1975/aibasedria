import tailwind from "@astrojs/tailwind";

/** @type {import('astro').AstroUserConfig} */
export default {
  base: "/regimpact-ai/",
  integrations: [tailwind({ applyBaseStyles: true })],
};

