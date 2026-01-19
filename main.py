import argparse
import os
import urllib.parse
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import requests
from dotenv import load_dotenv
from scipy import stats

load_dotenv()

# Configuration
IMMICH_URL = os.getenv("IMMICH_URL")
IMMICH_API_KEY = os.getenv("IMMICH_API_KEY")
NUM_CALLS = 100  # Number of times to call the API
ASSETS_PER_CALL = 250
INCLUDE_STACKED = True


def fetch_tag_id(headers, tag_name) -> str | None:
    """Fetch the tag ID from Immich"""
    if not IMMICH_URL:
        print("IMMICH_URL environment variable is not set.")
        return None

    try:
        print("Fetching tags")
        response = requests.get(
            urllib.parse.urljoin(IMMICH_URL, "api/tags"),
            headers=headers,
        )
        response.raise_for_status()
        tags = response.json()

        for tag in tags:
            if tag["value"] == tag_name:
                print(f"Found tag '{tag_name}' with ID {tag['id']}")
                return tag["id"]

        print(f"Tag '{tag_name}' not found")
        return None

    except requests.exceptions.RequestException as e:
        print(f"Error fetching tags: {e}")
        return None


def fetch_timeline_buckets(headers, **kwargs):
    """Fetch the timeline buckets to understand actual asset distribution"""
    if not IMMICH_URL:
        print("IMMICH_URL environment variable is not set.")
        return None, 0

    params = {
        "withStacked": "true" if INCLUDE_STACKED else "false",
    }

    tag_id = kwargs.get("tag_id", None)

    if tag_id:
        params["tagId"] = tag_id

    try:
        print("Fetching actual asset distribution from timeline buckets...")
        response = requests.get(
            urllib.parse.urljoin(IMMICH_URL, "api/timeline/buckets"),
            headers=headers,
            params=params,
        )
        response.raise_for_status()
        buckets = response.json()

        # Convert to year-based distribution
        year_distribution = {}
        for bucket in buckets:
            year = int(bucket["timeBucket"][:4])
            count = bucket["count"]
            year_distribution[year] = year_distribution.get(year, 0) + count

        total_assets = sum(year_distribution.values())
        print(f"Total assets in library: {total_assets}")
        print(
            f"Year range: {min(year_distribution.keys())} to {max(year_distribution.keys())}"
        )

        return year_distribution, total_assets
    except requests.exceptions.RequestException as e:
        print(f"Error fetching timeline buckets: {e}")
        return None, 0


def fetch_random_assets(headers, **kwargs):
    """Fetch random assets from the API"""
    if not IMMICH_URL:
        print("IMMICH_URL environment variable is not set.")
        return None, 0

    body = {"limit": ASSETS_PER_CALL, "withStacked": INCLUDE_STACKED}

    tag_id = kwargs.get("tag_id", None)
    if tag_id:
        body["tagIds"] = [tag_id]

    try:
        response = requests.post(
            urllib.parse.urljoin(IMMICH_URL, "api/search/random"),
            headers=headers,
            json=body,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching assets: {e}")
        return []


def collect_random_samples(num_calls, headers, **kwargs):
    """Collect random samples from multiple API calls"""
    all_samples = []
    sample_ids_with_duplicates = []  # Track all including duplicates

    for i in range(num_calls):
        print(f"Fetching random batch {i + 1}/{num_calls}...")
        assets = fetch_random_assets(headers, **kwargs)

        for asset in assets:
            all_samples.append(asset)
            sample_ids_with_duplicates.append(asset["id"])

        unique_count = len(set(sample_ids_with_duplicates))
        duplicate_rate = (1 - unique_count / len(sample_ids_with_duplicates)) * 100
        print(
            f"  Retrieved {len(assets)} assets, {unique_count} unique so far ({duplicate_rate:.1f}% duplicates)"
        )

    return all_samples


def extract_dates_from_assets(assets):
    """Extract valid dates from assets"""
    dates = []
    skipped_count = 0

    for asset in assets:
        date_str = asset.get("fileCreatedAt") or asset.get("createdAt")
        if date_str:
            try:
                dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                if 1677 <= dt.year <= 2262:
                    dates.append(dt)
                else:
                    skipped_count += 1
            except (ValueError, OverflowError):
                skipped_count += 1

    if skipped_count > 0:
        print(f"\nWarning: Skipped {skipped_count} assets with invalid dates")

    return dates


def analyze_recency_bias_with_baseline(
    sample_assets, actual_distribution, total_library_assets, with_tag=False
):
    """Compare random samples against actual library distribution"""

    # Extract dates from samples
    sample_dates = extract_dates_from_assets(sample_assets)

    if not sample_dates:
        print("No valid dates found in samples!")
        return None

    # Create DataFrames
    sample_df = pd.DataFrame({"date": pd.to_datetime(sample_dates, utc=True)})
    sample_df["year"] = sample_df["date"].dt.year

    print("\n" + "=" * 80)
    print("RECENCY BIAS ANALYSIS - COMPARING SAMPLE VS ACTUAL DISTRIBUTION")
    print("=" * 80)

    # Get sample distribution by year
    sample_year_counts = sample_df["year"].value_counts().to_dict()

    # Align years between actual and sample
    all_years = sorted(set(actual_distribution.keys()) | set(sample_year_counts.keys()))

    # Build comparison table
    comparison_data = []
    for year in all_years:
        actual_count = actual_distribution.get(year, 0)
        actual_pct = (
            (actual_count / total_library_assets) * 100
            if total_library_assets > 0
            else 0
        )

        sample_count = sample_year_counts.get(year, 0)
        sample_pct = (sample_count / len(sample_df)) * 100 if len(sample_df) > 0 else 0

        expected_in_sample = (actual_count / total_library_assets) * len(sample_df)

        # Calculate difference
        difference = sample_pct - actual_pct

        comparison_data.append(
            {
                "year": year,
                "actual_count": actual_count,
                "actual_pct": actual_pct,
                "sample_count": sample_count,
                "sample_pct": sample_pct,
                "expected_count": expected_in_sample,
                "difference_pct": difference,
            }
        )

    comparison_df = pd.DataFrame(comparison_data)

    print(f"\nLibrary: {total_library_assets:,} total assets")
    print(f"Sample: {len(sample_df):,} total assets")
    print(
        f"Unique samples: {sample_df['year'].nunique()} (from {len(sample_assets):,} API calls)"
    )

    print(
        f"\n{'Year':>7} {'Library %':>13} {'Sample %':>12} {'Difference':>12} {'Expected':>10} {'Actual':>10}"
    )

    print("-" * 80)

    for _, row in comparison_df.iterrows():
        year = int(row["year"])
        lib_pct = row["actual_pct"]
        samp_pct = row["sample_pct"]
        diff = row["difference_pct"]
        expected = row["expected_count"]
        actual = row["sample_count"]

        # Highlight significant differences
        marker = "⚠️  " if abs(diff) > 2 else "   "

        print(
            f"{marker}{year:<6} {lib_pct:>10.2f}%  {samp_pct:>10.2f}%  {diff:>+10.2f}%  {expected:>9.1f}  {actual:>9.0f}"
        )

    # Chi-square test for statistical significance
    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE TEST (Chi-Square)")
    print("=" * 80)

    observed = [sample_year_counts.get(year, 0) for year in all_years]
    expected = [
        (actual_distribution.get(year, 0) / total_library_assets) * len(sample_df)
        for year in all_years
    ]

    # Filter out years with very low expected counts
    filtered_observed = []
    filtered_expected = []
    filtered_years = []
    for year, obs, exp in zip(all_years, observed, expected):
        if exp >= 5:  # Chi-square requires expected >= 5
            filtered_observed.append(obs)
            filtered_expected.append(exp)
            filtered_years.append(year)

    if len(filtered_observed) > 1:
        # Normalize expected to match observed sum exactly (fixes floating point precision)
        observed_sum = sum(filtered_observed)
        expected_sum = sum(filtered_expected)
        filtered_expected = [
            e * (observed_sum / expected_sum) for e in filtered_expected
        ]

        chi2, p_value = stats.chisquare(filtered_observed, filtered_expected)
        print(f"Chi-square statistic: {chi2:.2f}")
        print(f"P-value: {p_value:.6f}")
        print(f"Degrees of freedom: {len(filtered_observed) - 1}")
        print(
            f"Years included in test: {len(filtered_years)} (excluded {len(all_years) - len(filtered_years)} with expected < 5)"
        )

        if p_value < 0.001:
            print("\n🔴 HIGHLY SIGNIFICANT bias detected (p < 0.001)")
            print("   The random selection is NOT uniformly random.")
        elif p_value < 0.05:
            print("\n🟡 SIGNIFICANT bias detected (p < 0.05)")
            print("   The random selection shows statistically significant deviation.")
        else:
            print("\n🟢 NO significant bias detected (p >= 0.05)")
            print("   The random selection appears reasonably uniform.")
    else:
        print("Not enough data for chi-square test")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Find years with biggest over/under representation
    comparison_df_sorted = comparison_df.sort_values("difference_pct", ascending=False)

    print("\nMost OVER-represented years:")
    over_count = 0
    for _, row in comparison_df_sorted.head(5).iterrows():
        if row["difference_pct"] > 0 and over_count < 3:
            print(
                f"  {int(row['year'])}: +{row['difference_pct']:.2f}% ({row['sample_count']:.0f} vs {row['expected_count']:.1f} expected)"
            )
            over_count += 1

    print("\nMost UNDER-represented years:")
    under_count = 0
    for _, row in comparison_df_sorted.tail(5).iterrows():
        if row["difference_pct"] < 0 and under_count < 3:
            print(
                f"  {int(row['year'])}: {row['difference_pct']:.2f}% ({row['sample_count']:.0f} vs {row['expected_count']:.1f} expected)"
            )
            under_count += 1

    # Create visualizations
    create_comparison_visualizations(comparison_df, sample_df, with_tag)

    return comparison_df


def create_comparison_visualizations(comparison_df, sample_df, with_tag=False):
    """Create visualizations comparing sample vs actual distribution"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    years = comparison_df["year"].values

    # 1. Side-by-side bar chart
    x = range(len(years))
    width = 0.35

    axes[0, 0].bar(
        [i - width / 2 for i in x],
        comparison_df["actual_pct"],
        width,
        label="Library Distribution",
        alpha=0.8,
    )
    axes[0, 0].bar(
        [i + width / 2 for i in x],
        comparison_df["sample_pct"],
        width,
        label="Sample Distribution",
        alpha=0.8,
    )
    axes[0, 0].set_xlabel("Year")
    axes[0, 0].set_ylabel("Percentage")
    axes[0, 0].set_title(
        f"Library vs Sample Distribution by Year{' (tag)' if with_tag else ''}"
    )
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([str(int(y)) for y in years], rotation=45, ha="right")
    axes[0, 0].legend()
    axes[0, 0].grid(axis="y", alpha=0.3)

    # 2. Difference plot
    colors = ["red" if d > 0 else "blue" for d in comparison_df["difference_pct"]]
    axes[0, 1].bar(x, comparison_df["difference_pct"], color=colors, alpha=0.6)
    axes[0, 1].axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    axes[0, 1].set_xlabel("Year")
    axes[0, 1].set_ylabel("Difference (Sample % - Library %)")
    axes[0, 1].set_title(
        f"Over/Under Representation by Year{' (tag)' if with_tag else ''}"
    )
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([str(int(y)) for y in years], rotation=45, ha="right")
    axes[0, 1].grid(axis="y", alpha=0.3)

    # 3. Expected vs Actual scatter
    axes[1, 0].scatter(
        comparison_df["expected_count"], comparison_df["sample_count"], alpha=0.6, s=100
    )

    # Add perfect correlation line
    max_val = max(
        comparison_df["expected_count"].max(), comparison_df["sample_count"].max()
    )
    axes[1, 0].plot([0, max_val], [0, max_val], "r--", alpha=0.5, label="Perfect match")

    axes[1, 0].set_xlabel("Expected Count (based on library)")
    axes[1, 0].set_ylabel("Actual Sample Count")
    axes[1, 0].set_title(
        f"Expected vs Actual Sample Counts{' (tag)' if with_tag else ''}"
    )
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # 4. Timeline distribution
    axes[1, 1].hist(sample_df["date"], bins=50, alpha=0.7, edgecolor="black")
    axes[1, 1].set_xlabel("Date")
    axes[1, 1].set_ylabel("Number of Assets")
    axes[1, 1].set_title(f"Sample Timeline Distribution{' (tag)' if with_tag else ''}")
    axes[1, 1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig("recency_bias_comparison.png", dpi=300, bbox_inches="tight")
    print("\nVisualization saved as 'recency_bias_comparison.png'")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="A test for Immich's `search/random` api endpoint"
    )

    parser.add_argument(
        "-t",
        "--tag",
        help="Tag name (human-readable value) from Immich",
    )

    args = parser.parse_args()

    print(
        f"Starting Immich Random Assets Recency Bias Test{' with tag' if args.tag else ''}"
    )
    print("=" * 80)

    headers = {"x-api-key": IMMICH_API_KEY, "Accept": "application/json"}

    # Step 1: Get actual distribution from library
    if args.tag:
        tag_id = fetch_tag_id(headers, args.tag)
        if not tag_id:
            print("Failed to fetch tag ID. Exiting.")
            return
        actual_distribution, total_assets = fetch_timeline_buckets(
            headers, tag_id=tag_id
        )
    else:
        actual_distribution, total_assets = fetch_timeline_buckets(headers)

    if not actual_distribution:
        print("Failed to fetch actual distribution. Exiting.")
        return

    # Step 2: Collect random samples
    print(f"\nCollecting {NUM_CALLS} random samples...")
    if args.tag:
        sample_assets = collect_random_samples(NUM_CALLS, headers, tag_id=tag_id)
    else:
        sample_assets = collect_random_samples(NUM_CALLS, headers)

    if not sample_assets:
        print("No samples collected. Please check your API endpoint and credentials.")
        return

    # Step 3: Analyze for recency bias
    with_tag = bool(args.tag)
    analyze_recency_bias_with_baseline(
        sample_assets, actual_distribution, total_assets, with_tag
    )

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
