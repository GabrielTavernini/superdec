import pandas as pd
import argparse
import sys


def main():
  parser = argparse.ArgumentParser(description='Select best rows by a metric and plot means of other metrics.')
  parser.add_argument('csvs', nargs='+', help='CSV files to read and concatenate')
  parser.add_argument('--metric', default='chamfer-L1', help='Metric to use for selecting best per group')
  parser.add_argument('--func', choices=['min', 'max'], default='min', help='Use min or max to pick best')
  parser.add_argument('--group-by', default='index', help='Column name to group by when selecting best')
  # no plotting: this script will print means to stdout
  args = parser.parse_args()

  try:
    df = pd.concat([pd.read_csv(f) for f in args.csvs], ignore_index=True)
  except Exception as e:
    print('Error reading CSV files:', e, file=sys.stderr)
    sys.exit(1)

  if args.group_by not in df.columns:
    print(f"Group-by column '{args.group_by}' not found in data", file=sys.stderr)
    sys.exit(1)

  if args.metric not in df.columns:
    print(f"Selection metric '{args.metric}' not found in data", file=sys.stderr)
    sys.exit(1)

  grp = df.groupby(args.group_by)
  if args.func == 'min':
    idx = grp[args.metric].idxmin()
  else:
    idx = grp[args.metric].idxmax()

  selected = df.loc[idx].reset_index(drop=True)

  numeric_cols = selected.select_dtypes(include='number').columns.tolist()
  # remove grouping and selection columns from the plotted metrics
  numeric_cols = [c for c in numeric_cols if c != args.group_by]

  if not numeric_cols:
    print('No numeric metrics found to compute means on.', file=sys.stderr)
    sys.exit(1)

  means = selected[numeric_cols].mean()

  means = means.sort_values()
  print(f"Means of metrics (selected by {args.func} {args.metric}):")
  print(means.to_string())


if __name__ == '__main__':
  main()