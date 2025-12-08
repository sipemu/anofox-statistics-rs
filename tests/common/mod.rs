use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;

/// Load a single-row CSV of scalar reference values into a HashMap.
/// Each column becomes a key, its value becomes the f64 value.
pub fn load_reference_scalars(filename: &str) -> HashMap<String, f64> {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("R/data");
    path.push(filename);

    let file = File::open(&path).unwrap_or_else(|e| {
        panic!(
            "Could not open reference file '{}': {}. Did you run 'Rscript R/generate_refs.R'?",
            path.display(),
            e
        )
    });

    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(file);

    let headers = rdr.headers().unwrap().clone();
    let mut result = HashMap::new();

    if let Some(record_result) = rdr.records().next() {
        let record = record_result.unwrap();
        for (i, field) in record.iter().enumerate() {
            let value: f64 = field
                .parse()
                .unwrap_or_else(|e| panic!("Could not parse '{}' as f64: {}", field, e));
            result.insert(headers[i].to_string(), value);
        }
    }

    result
}

/// Load a single-column CSV of vector values.
pub fn load_reference_vector(filename: &str) -> Vec<f64> {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("R/data");
    path.push(filename);

    let file = File::open(&path).unwrap_or_else(|e| {
        panic!(
            "Could not open reference file '{}': {}. Did you run 'Rscript R/generate_refs.R'?",
            path.display(),
            e
        )
    });

    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(file);

    let mut result = Vec::new();

    for record_result in rdr.records() {
        let record = record_result.unwrap();
        if let Some(field) = record.get(0) {
            let value: f64 = field
                .parse()
                .unwrap_or_else(|e| panic!("Could not parse '{}' as f64: {}", field, e));
            result.push(value);
        }
    }

    result
}
