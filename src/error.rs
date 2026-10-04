use std::fmt;

#[derive(Debug)]
pub enum Error {
    /// No GPU adapter matched (no driver, or running somewhere without a GPU).
    NoAdapter(String),
    /// The adapter refused our device request (usually an unsupported limit or feature).
    Device(String),
    /// Shapes don't fit the operation, or data length doesn't match the shape.
    Shape(String),
    /// Reading a buffer back from the GPU failed.
    Readback(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NoAdapter(e) => write!(f, "no GPU adapter: {e}"),
            Error::Device(e) => write!(f, "device request failed: {e}"),
            Error::Shape(e) => write!(f, "shape error: {e}"),
            Error::Readback(e) => write!(f, "readback failed: {e}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
