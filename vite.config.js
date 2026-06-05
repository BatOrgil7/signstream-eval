import { defineConfig } from "vite";

export default defineConfig({
  cacheDir: process.env.VITE_CACHE_DIR || ".vite-cache",
  server: {
    fs: {
      strict: true,
    },
  },
});
