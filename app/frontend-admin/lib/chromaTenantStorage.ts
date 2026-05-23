/**
 * @deprecated Use `@/lib/vectorDbConnectionConfig` for Pipeline Builder.
 * Re-exports kept for backward compatibility.
 */
export {
  DEFAULT_COLLECTION as DEFAULT_CHROMA_COLLECTION,
  FORBIDDEN_CHROMA_PATHS,
  defaultChromaPersistPath,
  validateChromaPersistPath,
} from "./vectorDbConnectionConfig";
