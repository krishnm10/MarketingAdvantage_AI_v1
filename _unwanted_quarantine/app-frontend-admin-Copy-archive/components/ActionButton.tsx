"use client";

type ActionButtonProps = {
  label: string;
  onClick: () => Promise<void>;
  variant?: "primary" | "danger";
};

export default function ActionButton({
  label,
  onClick,
  variant = "primary",
}: ActionButtonProps) {
  const base =
    "rounded px-4 py-2 text-sm font-medium transition disabled:opacity-50";
  const styles =
    variant === "danger"
      ? "bg-red-600 text-white hover:bg-red-700"
      : "bg-blue-600 text-white hover:bg-blue-700";

  return (
    <button
      className={`${base} ${styles}`}
      onClick={onClick}
      type="button"
    >
      {label}
    </button>
  );
}
