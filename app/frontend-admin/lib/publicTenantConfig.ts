/**
 * TypeScript mirror of backend PublicTenantConfig (Phase 6).
 * Secret-free — safe to hold in browser memory.
 */

export interface PublicProviderFlags {
  embedder_type: string;
  embedder_configured: boolean;
  llm_configured: boolean;
  llm_provider: string | null;
  llm_model: string | null;
  vectordb_type: string;
  reranker_enabled: boolean;
  reranker_type: string | null;
  secret_store_provider: string | null;
}

export interface PublicVisionConfig {
  ai_profile: string;
  vision_model_cpu: string;
  vision_model_gpu: string;
  vision_api_provider: string;
  vision_api_model: string;
  vision_api_configured: boolean;
  vision_quantize: string;
  vision_flash_attention: boolean;
  enable_visual_explanation: boolean;
  enable_audio_language_detection: boolean;
  video_vision_frames: number;
}

export interface PublicIngestionConfig {
  enable_visual_llm_explanation: boolean;
  visual_llm_concurrency: number;
  vision: PublicVisionConfig;
}

export interface FeatureFlags {
  enable_rag: boolean;
  enable_reranking: boolean;
  enable_hybrid_search: boolean;
  enable_multi_tenant_isolation: boolean;
  enable_audit_log: boolean;
  enable_realtime_ingestion: boolean;
  enable_scheduler_validation: boolean;
  enable_conflict_detection: boolean;
  enable_temporal_validation: boolean;
  enable_websocket_progress: boolean;
  max_concurrent_ingestions: number;
  enable_bundle_validation: boolean;
  enable_pii_middleware: boolean;
  enable_advanced_nodes: boolean;
  enable_l1_task_classification: boolean;
  enable_docset_analysis: boolean;
  enable_docset_analysis_debug: boolean;
}

export interface ParserConfig {
  enable_pdf: boolean;
  enable_docx: boolean;
  enable_xlsx: boolean;
  enable_csv: boolean;
  enable_pptx: boolean;
  enable_html: boolean;
  enable_json: boolean;
  enable_txt: boolean;
  enable_ocr: boolean;
  enable_audio: boolean;
  enable_video: boolean;
  enable_image: boolean;
}

export interface PublicTenantConfig {
  client_id: string;
  client_name: string;
  description: string;
  version: string;
  providers: PublicProviderFlags;
  features: FeatureFlags;
  parsers: ParserConfig;
  ingestion: PublicIngestionConfig;
}

export function isFeatureEnabled(
  config: PublicTenantConfig | null | undefined,
  flag: keyof FeatureFlags
): boolean {
  if (!config?.features) return false;
  const val = config.features[flag];
  return typeof val === "boolean" ? val : Boolean(val);
}
