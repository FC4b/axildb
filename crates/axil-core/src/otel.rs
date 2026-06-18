//! OpenTelemetry instrumentation for Axil.
//!
//! Enabled via the `otel` feature flag. When disabled, all functions are
//! no-ops with zero overhead (empty inline functions).

#[cfg(feature = "otel")]
use opentelemetry::metrics::{Counter, Histogram};
#[cfg(feature = "otel")]
use opentelemetry::KeyValue;
#[cfg(feature = "otel")]
use std::sync::OnceLock;

#[cfg(feature = "otel")]
static METER: OnceLock<AxilMeter> = OnceLock::new();

#[cfg(feature = "otel")]
struct AxilMeter {
    op_counter: Counter<u64>,
    op_latency: Histogram<f64>,
    index_size: Histogram<u64>,
}

#[cfg(feature = "otel")]
fn meter() -> &'static AxilMeter {
    METER.get_or_init(|| {
        let meter = opentelemetry::global::meter("axil");
        AxilMeter {
            op_counter: meter
                .u64_counter("axil.operations")
                .with_description("Number of Axil operations")
                .build(),
            op_latency: meter
                .f64_histogram("axil.operation.duration")
                .with_description("Operation latency in milliseconds")
                .with_unit("ms")
                .build(),
            index_size: meter
                .u64_histogram("axil.index.size")
                .with_description("Index size (record count)")
                .build(),
        }
    })
}

/// Initialize the OpenTelemetry pipeline with OTLP export.
///
/// Sets the `OTEL_EXPORTER_OTLP_ENDPOINT` env var and configures both
/// trace and metrics pipelines. Returns a guard that shuts down on drop.
#[cfg(feature = "otel")]
pub fn init_otel(endpoint: &str) -> Result<OtelGuard, String> {
    use opentelemetry_otlp::{MetricExporter, SpanExporter, WithExportConfig};
    use opentelemetry_sdk::trace::SdkTracerProvider;
    use opentelemetry_sdk::Resource;

    let resource = Resource::builder().with_service_name("axil").build();

    // Trace exporter via gRPC/tonic
    let exporter = SpanExporter::builder()
        .with_tonic()
        .with_endpoint(endpoint)
        .build()
        .map_err(|e| format!("failed to create OTLP span exporter: {e}"))?;

    let provider = SdkTracerProvider::builder()
        .with_resource(resource.clone())
        .with_batch_exporter(exporter)
        .build();

    opentelemetry::global::set_tracer_provider(provider.clone());

    // Metrics exporter via gRPC/tonic
    let metrics_exporter = MetricExporter::builder()
        .with_tonic()
        .with_endpoint(endpoint)
        .build()
        .map_err(|e| format!("failed to create metrics exporter: {e}"))?;

    let reader = opentelemetry_sdk::metrics::PeriodicReader::builder(metrics_exporter).build();

    let metrics_provider = opentelemetry_sdk::metrics::SdkMeterProvider::builder()
        .with_resource(resource)
        .with_reader(reader)
        .build();

    opentelemetry::global::set_meter_provider(metrics_provider.clone());

    Ok(OtelGuard {
        _tracer_provider: provider,
        _meter_provider: metrics_provider,
    })
}

/// Guard that shuts down the OpenTelemetry pipeline on drop.
#[cfg(feature = "otel")]
pub struct OtelGuard {
    _tracer_provider: opentelemetry_sdk::trace::SdkTracerProvider,
    _meter_provider: opentelemetry_sdk::metrics::SdkMeterProvider,
}

/// No-op guard when OTel is disabled.
#[cfg(not(feature = "otel"))]
pub struct OtelGuard;

/// No-op init when OTel is disabled.
#[cfg(not(feature = "otel"))]
pub fn init_otel(_endpoint: &str) -> Result<OtelGuard, String> {
    Ok(OtelGuard)
}

// ── Instrumentation helpers ───────────────────────────────────────────

/// Record an operation for OTel metrics.
#[cfg(feature = "otel")]
pub fn record_operation(op: &str, table: &str, latency_ms: f64) {
    let m = meter();
    let attrs = [
        KeyValue::new("op", op.to_string()),
        KeyValue::new("table", table.to_string()),
    ];
    m.op_counter.add(1, &attrs);
    m.op_latency.record(latency_ms, &attrs);
}

#[cfg(not(feature = "otel"))]
#[inline(always)]
pub fn record_operation(_op: &str, _table: &str, _latency_ms: f64) {}

/// Record index size for OTel metrics.
#[cfg(feature = "otel")]
pub fn record_index_size(index_type: &str, count: u64) {
    let m = meter();
    m.index_size
        .record(count, &[KeyValue::new("index", index_type.to_string())]);
}

#[cfg(not(feature = "otel"))]
#[inline(always)]
pub fn record_index_size(_index_type: &str, _count: u64) {}

/// Start a trace span for an operation. Returns a guard that ends the span on drop.
#[cfg(feature = "otel")]
pub fn span(name: &'static str, attrs: &[(&'static str, String)]) -> SpanGuard {
    use opentelemetry::trace::{SpanKind, Tracer};

    let tracer = opentelemetry::global::tracer("axil");

    let kv_attrs: Vec<KeyValue> = attrs
        .iter()
        .map(|(k, v)| KeyValue::new(*k, v.clone()))
        .collect();

    let span = tracer
        .span_builder(name)
        .with_kind(SpanKind::Internal)
        .with_attributes(kv_attrs)
        .start(&tracer);

    let guard = opentelemetry::trace::mark_span_as_active(span);

    SpanGuard { _guard: guard }
}

#[cfg(feature = "otel")]
pub struct SpanGuard {
    _guard: opentelemetry::ContextGuard,
}

/// No-op span when OTel is disabled.
#[cfg(not(feature = "otel"))]
#[inline(always)]
pub fn span(_name: &'static str, _attrs: &[(&'static str, String)]) -> SpanGuard {
    SpanGuard
}

#[cfg(not(feature = "otel"))]
pub struct SpanGuard;
