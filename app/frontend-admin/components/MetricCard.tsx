// frontend-admin/components/MetricCard.tsx

type MetricCardProps = {
  title: string;
  value: number | string;
};

export default function MetricCard({ title, value }: MetricCardProps) {
  return (
    <div className="rounded-xl border border-slate-200/60 bg-white p-5 shadow-card hover:shadow-card-hover transition-shadow duration-300">
      <div className="text-xs font-medium uppercase tracking-wider text-slate-400">{title}</div>
      <div className="mt-2 text-3xl font-bold text-slate-900">{value}</div>
    </div>
  );
}
