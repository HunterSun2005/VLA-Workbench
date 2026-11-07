# Removed components

Historical migration snapshot. The retained routes listed below describe the migration stage, not the current execution architecture; see [current module locations](FEATURES.md) and [execution contracts](../EXECUTION_RTC_RLT.md).

SO100/SO101 record package and its tests, disconnected legacy online_collector, experimental warmup visualizer and its tests, machine-specific experiment scripts, original bucket/cache builder and legacy AC CLI were excluded. The manifest-based v2 cache builder, native AC trainer, generic intervention blending, and all Franka collection routes remain. New default RLT mappings use Franka vector features; explicit configurations preserve existing dimensions.
