// frontend-admin/components/MetricCard.tsx

type MetricCardProps = {
  title: string;
  value: number | string;
};

export default function MetricCard({ title, value }: MetricCardProps) {
  return (
    <div className="rounded-lg border bg-white p-4 shadow-sm">
      <div className="text-sm text-gray-500">{title}</div>
      <div className="mt-2 text-2xl font-semibold">{value}</div>
    </div>
  );
}
