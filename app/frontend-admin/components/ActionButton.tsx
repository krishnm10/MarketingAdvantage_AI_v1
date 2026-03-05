"use client";

type ActionButtonProps = {
  label: string;
  onClick: () => Promise<void>;
  variant?: "primary" | "danger";
  disabled?: boolean;
};

export default function ActionButton({
  label,
  onClick,
  variant = "primary",
  disabled = false,
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
      disabled={disabled}
    >
      {label}
    </button>
  );
}
