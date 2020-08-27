use std::ops::{Deref, DerefMut};

#[derive(Copy, Clone, Debug)]
pub struct SpacetimeCoords([f64; 3]);

// is this a sin? it feels like this is bad style
impl Deref for SpacetimeCoords {
	type Target = [f64; 3];

	fn deref(&self) -> &Self::Target {
		&self.0
	}
}

impl DerefMut for SpacetimeCoords {
	fn deref_mut(&mut self) -> &mut Self::Target {
		&mut self.0
	}
}

impl SpacetimeCoords {
	// boy I sure love copy/pasting code
	// I should really figure out macros

	pub fn new(t: f64, r: f64, phi: f64) -> Self {
		SpacetimeCoords([t, r, phi])
	}

	pub fn t(&self) -> &f64 {
		&self[0]
	}

	pub fn r(&self) -> &f64 {
		&self[1]
	}

	pub fn phi(&self) -> &f64 {
		&self[2]
	}

	pub fn t_mut(&mut self) -> &mut f64 {
		&mut self[0]
	}

	pub fn r_mut(&mut self) -> &mut f64 {
		&mut self[1]
	}

	pub fn phi_mut(&mut self) -> &mut f64 {
		&mut self[2]
	}
}

#[derive(Copy, Clone, Debug)]
pub struct SphericalCoords {
	pub theta: f64,
	pub phi: f64,
}

#[derive(Copy, Clone, Debug)]
pub struct XYZUnitCoords {
	x: f64,
	y: f64,
	z: f64,
}

impl XYZUnitCoords {
	pub fn from_unnormed(x: f64, y: f64, z: f64) -> Self {
		let len = (x * x + y * y + z * z).sqrt();

		if len == 0.0 {
			panic!("attempted to construct XYZUnitCoords from zero vector");
		}

		XYZUnitCoords {
			x: x / len,
			y: y / len,
			z: z / len,
		}
	}

	pub unsafe fn new_unchecked(x: f64, y: f64, z: f64) -> Self {
		XYZUnitCoords {x, y, z}
	}

	pub fn x(&self) -> &f64 {
		&self.x
	}

	pub fn y(&self) -> &f64 {
		&self.y
	}

	pub fn z(&self) -> &f64 {
		&self.z
	}
}

impl From<SphericalCoords> for XYZUnitCoords {
	fn from(sphere_coords: SphericalCoords) -> XYZUnitCoords {
		let SphericalCoords {theta, phi} = sphere_coords;

		XYZUnitCoords {
			x: theta.sin() * phi.cos(),
			y: theta.cos(),
			z: theta.sin() * phi.sin(),
		}
	}
}

impl From<XYZUnitCoords> for SphericalCoords {
	fn from(xyz_coords: XYZUnitCoords) -> SphericalCoords {
		let XYZUnitCoords {x, y, z} = xyz_coords;

		SphericalCoords {
			theta: (x * x + z * z).sqrt().atan2(y),
			phi: z.atan2(x),
		}
	}
}

#[derive(Copy, Clone, Debug)]
pub struct XYZCoords {
	pub x: f64,
	pub y: f64,
	pub z: f64,
}

impl XYZCoords {
	pub fn to_unit_and_len(&self) -> Option<(XYZUnitCoords, f64)> {
		let XYZCoords {x, y, z} = self;
		let len = (x * x + y * y + z * z).sqrt();

		if len == 0.0 {
			return None;
		}

		let unit = XYZUnitCoords {
			x: x / len,
			y: y / len,
			z: z / len,
		};

		Some((unit, len))
	}

	pub fn dot(&self, other: &Self) -> f64 {
		self.x * other.x + self.y * other.y + self.z * other.z
	}
}

impl From<XYZUnitCoords> for XYZCoords {
	fn from(unit_coords: XYZUnitCoords) -> XYZCoords {
		let XYZUnitCoords {x, y, z} = unit_coords;

		XYZCoords {x, y, z}
	}
}
