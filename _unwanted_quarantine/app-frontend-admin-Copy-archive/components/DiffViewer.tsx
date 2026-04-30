"use client";

type DiffViewerProps = {
  before: string;
  after: string;
};

export default function DiffViewer({ before, after }: DiffViewerProps) {
  return (
    <div className="grid grid-cols-2 gap-4 text-sm">
      <div>
        <div className="mb-1 font-medium text-gray-600">Before</div>
        <pre className="whitespace-pre-wrap rounded border bg-gray-50 p-3">
          {before}
        </pre>
      </div>

      <div>
        <div className="mb-1 font-medium text-gray-600">After</div>
        <pre className="whitespace-pre-wrap rounded border bg-yellow-50 p-3">
          {after}
        </pre>
      </div>
    </div>
  );
}
