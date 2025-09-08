module.exports = {
  root: true,
  env: { browser: true, es2022: true },
  extends: ["eslint:recommended", "plugin:astro/recommended"],
  overrides: [
    {
      files: ["*.astro"],
      parser: "astro-eslint-parser",
      parserOptions: { parser: "@typescript-eslint/parser" }
    }
  ]
};

