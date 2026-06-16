"use client";

import { Info } from "lucide-react";
import Link from "next/link";
import { cn } from "@/lib/utils";
import {
  type VectorDbConnectionDraft,
  defaultChromaPersistPath,
} from "@/lib/vectorDbConnectionConfig";

interface TenantVectorDbConnectionFieldsProps {
  clientId: string;
  draft: VectorDbConnectionDraft;
  onChange: (next: VectorDbConnectionDraft) => void;
  validationError: string | null;
  isNewTenantOverlay?: boolean;
}

function Label({
  children,
  required,
}: {
  children: React.ReactNode;
  required?: boolean;
}) {
  return (
    <label className="text-xs font-semibold text-slate-600 block mb-1">
      {children}
      {required ? <span className="text-red-600"> *</span> : null}
    </label>
  );
}

function TextInput({
  value,
  onChange,
  placeholder,
  type = "text",
}: {
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  type?: string;
}) {
  return (
    <input
      type={type}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      className="w-full h-9 rounded-lg border border-slate-200 bg-white px-3 text-sm text-slate-800 focus:border-primary-400 focus:outline-none focus:ring-1 focus:ring-primary-200"
    />
  );
}

function ModeToggle<T extends string>({
  modes,
  value,
  onChange,
}: {
  modes: { id: T; label: string }[];
  value: T;
  onChange: (m: T) => void;
}) {
  return (
    <div className="flex rounded-lg border border-slate-200 p-0.5 bg-slate-50 w-fit">
      {modes.map((m) => (
        <button
          key={m.id}
          type="button"
          onClick={() => onChange(m.id)}
          className={cn(
            "px-3 py-1.5 text-xs font-medium rounded-md transition-colors",
            value === m.id
              ? "bg-white text-primary-700 shadow-sm border border-slate-200"
              : "text-slate-600 hover:text-slate-800"
          )}
        >
          {m.label}
        </button>
      ))}
    </div>
  );
}

function SecretRefField({
  label,
  value,
  onChange,
  placeholder,
  required,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  required?: boolean;
}) {
  return (
    <div>
      <Label required={required}>{label}</Label>
      <TextInput
        value={value}
        onChange={onChange}
        placeholder={placeholder ?? "e.g. env://QDRANT_API_KEY"}
      />
      <p className="text-[11px] text-slate-500 mt-1">
        SecretRef URI pointer stored in tenant JSON — not the raw secret value.
        Set the actual credential on Secrets &amp; Security.
      </p>
    </div>
  );
}

export function TenantVectorDbConnectionFields({
  clientId,
  draft,
  onChange,
  validationError,
  isNewTenantOverlay,
}: TenantVectorDbConnectionFieldsProps) {
  const providerLabel =
    draft.provider.charAt(0).toUpperCase() + draft.provider.slice(1);

  const renderProviderFields = () => {
    switch (draft.provider) {
      case "chroma":
        return (
          <>
            <ModeToggle
              modes={[
                { id: "local" as const, label: "Local (on-disk)" },
                { id: "remote" as const, label: "Remote / cloud" },
              ]}
              value={draft.chromaMode}
              onChange={(chromaMode) => onChange({ ...draft, chromaMode })}
            />
            {draft.chromaMode === "local" ? (
              <div>
                <Label required>Chroma persist directory</Label>
                <TextInput
                  value={draft.persistDirectory}
                  onChange={(persistDirectory) => onChange({ ...draft, persistDirectory })}
                  placeholder={defaultChromaPersistPath(clientId)}
                />
                <p className="text-[11px] text-slate-500 mt-1">
                  Each tenant needs its own directory — never share with another tenant or{" "}
                  <code className="text-[10px]">default</code>.
                </p>
              </div>
            ) : (
              <>
                <div>
                  <Label required>Hostname</Label>
                  <TextInput
                    value={draft.chromaHost}
                    onChange={(chromaHost) => onChange({ ...draft, chromaHost })}
                    placeholder="e.g. chromadb.internal"
                  />
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <Label>Port</Label>
                    <TextInput
                      value={draft.chromaPort}
                      onChange={(chromaPort) => onChange({ ...draft, chromaPort })}
                      placeholder="8000"
                    />
                  </div>
                  <div className="flex items-end pb-1">
                    <label className="flex items-center gap-2 text-xs text-slate-600 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={draft.chromaSsl}
                        onChange={(e) =>
                          onChange({ ...draft, chromaSsl: e.target.checked })
                        }
                        className="rounded border-slate-300"
                      />
                      Use SSL/TLS
                    </label>
                  </div>
                </div>
                <SecretRefField
                  label="secret_ref URI (optional)"
                  value={draft.chromaApiKeyEnv}
                  onChange={(chromaApiKeyEnv) => onChange({ ...draft, chromaApiKeyEnv })}
                  placeholder="env://CHROMA_API_KEY"
                />
              </>
            )}
          </>
        );

      case "qdrant":
        return (
          <>
            <ModeToggle
              modes={[
                { id: "local" as const, label: "Local" },
                { id: "cloud" as const, label: "Cloud (URL)" },
              ]}
              value={draft.qdrantMode}
              onChange={(qdrantMode) => onChange({ ...draft, qdrantMode })}
            />
            {draft.qdrantMode === "cloud" ? (
              <>
                <div>
                  <Label required>Cluster URL</Label>
                  <TextInput
                    value={draft.qdrantUrl}
                    onChange={(qdrantUrl) => onChange({ ...draft, qdrantUrl })}
                    placeholder="https://xxx.qdrant.io"
                  />
                </div>
                <SecretRefField
                  label="secret_ref URI"
                  value={draft.qdrantApiKeyEnv}
                  onChange={(qdrantApiKeyEnv) => onChange({ ...draft, qdrantApiKeyEnv })}
                  placeholder="env://QDRANT_API_KEY"
                />
              </>
            ) : (
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <Label required>Host</Label>
                  <TextInput
                    value={draft.qdrantHost}
                    onChange={(qdrantHost) => onChange({ ...draft, qdrantHost })}
                    placeholder="localhost"
                  />
                </div>
                <div>
                  <Label>Port</Label>
                  <TextInput
                    value={draft.qdrantPort}
                    onChange={(qdrantPort) => onChange({ ...draft, qdrantPort })}
                    placeholder="6333"
                  />
                </div>
              </div>
            )}
          </>
        );

      case "pinecone":
        return (
          <>
            <ModeToggle
              modes={[
                { id: "cloud" as const, label: "Cloud" },
                { id: "local" as const, label: "Local (dev)" },
              ]}
              value={draft.pineconeMode}
              onChange={(pineconeMode) => onChange({ ...draft, pineconeMode })}
            />
            <div>
              <Label required>Index name</Label>
              <TextInput
                value={draft.pineconeIndexName}
                onChange={(pineconeIndexName) => onChange({ ...draft, pineconeIndexName })}
              />
            </div>
            <div>
              <Label>Namespace</Label>
              <TextInput
                value={draft.pineconeNamespace}
                onChange={(pineconeNamespace) => onChange({ ...draft, pineconeNamespace })}
                placeholder="default"
              />
            </div>
            <div>
              <Label>Embedding dimension</Label>
              <TextInput
                value={draft.pineconeEmbeddingDim}
                onChange={(pineconeEmbeddingDim) =>
                  onChange({ ...draft, pineconeEmbeddingDim })
                }
                placeholder="768"
              />
            </div>
            {draft.pineconeMode === "cloud" ? (
              <SecretRefField
                label="secret_ref URI"
                value={draft.pineconeApiKeyEnv}
                onChange={(pineconeApiKeyEnv) => onChange({ ...draft, pineconeApiKeyEnv })}
                placeholder="env://PINECONE_API_KEY"
                required
              />
            ) : (
              <div>
                <Label required>Local index path</Label>
                <TextInput
                  value={draft.pineconeLocalPath}
                  onChange={(pineconeLocalPath) => onChange({ ...draft, pineconeLocalPath })}
                  placeholder="./pinecone_local"
                />
              </div>
            )}
          </>
        );

      case "weaviate":
        return (
          <>
            <div>
              <Label required>Weaviate URL</Label>
              <TextInput
                value={draft.weaviateUrl}
                onChange={(weaviateUrl) => onChange({ ...draft, weaviateUrl })}
                placeholder="http://localhost:8080 or WCS URL"
              />
            </div>
            <SecretRefField
              label="secret_ref URI (WCS / auth)"
              value={draft.weaviateApiKeyEnv}
              onChange={(weaviateApiKeyEnv) => onChange({ ...draft, weaviateApiKeyEnv })}
              placeholder="env://WEAVIATE_API_KEY"
            />
          </>
        );

      case "milvus":
        return (
          <>
            <ModeToggle
              modes={[
                { id: "local" as const, label: "Local" },
                { id: "cloud" as const, label: "Zilliz Cloud (URI)" },
              ]}
              value={draft.milvusMode}
              onChange={(milvusMode) => onChange({ ...draft, milvusMode })}
            />
            {draft.milvusMode === "cloud" ? (
              <>
                <div>
                  <Label required>URI</Label>
                  <TextInput
                    value={draft.milvusUri}
                    onChange={(milvusUri) => onChange({ ...draft, milvusUri })}
                    placeholder="https://xxx.zillizcloud.com"
                  />
                </div>
                <SecretRefField
                  label="secret_ref URI"
                  value={draft.milvusTokenEnv}
                  onChange={(milvusTokenEnv) => onChange({ ...draft, milvusTokenEnv })}
                  placeholder="env://MILVUS_TOKEN"
                />
              </>
            ) : (
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <Label required>Host</Label>
                  <TextInput
                    value={draft.milvusHost}
                    onChange={(milvusHost) => onChange({ ...draft, milvusHost })}
                    placeholder="localhost"
                  />
                </div>
                <div>
                  <Label>Port</Label>
                  <TextInput
                    value={draft.milvusPort}
                    onChange={(milvusPort) => onChange({ ...draft, milvusPort })}
                    placeholder="19530"
                  />
                </div>
              </div>
            )}
          </>
        );

      case "redis":
        return (
          <>
            <ModeToggle
              modes={[
                { id: "local" as const, label: "Local" },
                { id: "cloud" as const, label: "Cloud (URL)" },
              ]}
              value={draft.redisMode}
              onChange={(redisMode) => onChange({ ...draft, redisMode })}
            />
            {draft.redisMode === "cloud" ? (
              <>
                <div>
                  <Label required>Redis URL</Label>
                  <TextInput
                    value={draft.redisUrl}
                    onChange={(redisUrl) => onChange({ ...draft, redisUrl })}
                    placeholder="rediss://..."
                  />
                </div>
                <SecretRefField
                  label="secret_ref URI"
                  value={draft.redisPasswordEnv}
                  onChange={(redisPasswordEnv) => onChange({ ...draft, redisPasswordEnv })}
                  placeholder="env://REDIS_PASSWORD"
                />
              </>
            ) : (
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <Label required>Host</Label>
                  <TextInput
                    value={draft.redisHost}
                    onChange={(redisHost) => onChange({ ...draft, redisHost })}
                    placeholder="localhost"
                  />
                </div>
                <div>
                  <Label>Port</Label>
                  <TextInput
                    value={draft.redisPort}
                    onChange={(redisPort) => onChange({ ...draft, redisPort })}
                    placeholder="6379"
                  />
                </div>
              </div>
            )}
            <label className="flex items-center gap-2 text-xs text-slate-600 cursor-pointer">
              <input
                type="checkbox"
                checked={draft.redisSsl}
                onChange={(e) => onChange({ ...draft, redisSsl: e.target.checked })}
                className="rounded border-slate-300"
              />
              Use SSL/TLS
            </label>
          </>
        );

      default:
        return null;
    }
  };

  return (
    <div className="mt-4 rounded-xl border border-violet-200 bg-violet-50/30 p-4 space-y-4">
      <div className="flex items-start gap-2">
        <Info className="h-4 w-4 text-violet-700 mt-0.5 flex-shrink-0" />
        <div>
          <p className="text-sm font-semibold text-slate-800">
            {providerLabel} connection (per tenant)
          </p>
          <p className="text-xs text-slate-600 mt-0.5">
            Choose local or remote/cloud mode, then set connection details stored in Client JSON.
            {isNewTenantOverlay
              ? " First apply creates this tenant’s config file."
              : null}
          </p>
        </div>
      </div>

      <div>
        <Label required>Collection name</Label>
        <TextInput
          value={draft.collection}
          onChange={(collection) => onChange({ ...draft, collection })}
          placeholder="ingested_content"
        />
        <p className="text-[11px] text-slate-500 mt-1">
          Logical collection / index name for this tenant inside the vector store.
        </p>
      </div>

      {renderProviderFields()}

      {validationError ? (
        <p className="text-xs text-amber-800 rounded-lg border border-amber-200 bg-amber-50 px-2 py-1.5">
          {validationError}
        </p>
      ) : null}

      <p className="text-[11px] text-slate-500 border-t border-violet-100 pt-2">
        Credentials are referenced by <code className="font-mono text-[10px]">secret_ref</code> URI in
        tenant JSON; set the actual secret values on{" "}
        <Link href="/secrets" className="text-primary-700 underline-offset-2 hover:underline">
          Secrets &amp; Security
        </Link>
        .
      </p>
    </div>
  );
}
