mod coords;
mod tensor;

use std::f64::consts::{PI, TAU};
use std::time::Instant;

use image::{ImageBuffer, Rgb, Rgba, open};

use coords::{
	SpacetimeCoords,
	SphericalCoords,
	XYZCoords,
	XYZUnitCoords,
};
use tensor::{Tensor, Index2, Index3, NUM_DIMS};

const WIDTH: u32 = 400;
const HEIGHT: u32 = 400;
const FOV_DEGREES: f64 = 45.0;

fn main() {
	let start = Instant::now();

	if WIDTH != HEIGHT {
		panic!("only support square images right now");
	}

	let fov_multiplier = (FOV_DEGREES / 360.0 * TAU / 2.0).tan();

	let mut image = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(WIDTH, HEIGHT);

	let background_dynamic = open("./background.png").unwrap();

	let background = background_dynamic.as_rgba8().unwrap();

	let camera = Camera {
		camera_position: XYZCoords {x: 3.5, y: 0.0, z: 0.0},
		camera_facing: XYZUnitCoords::from_unnormed(-1.0, 0.0, 0.0),
		camera_right: XYZUnitCoords::from_unnormed(0.0, 0.0, 1.0),
		camera_up: XYZUnitCoords::from_unnormed(0.0, 1.0, 0.0),
	};

	let delta_t = 0.01;

	for x in 0 .. WIDTH {
		let x_scale = x as f64 / WIDTH as f64 * 2.0 - 1.0;

		for y in 0 .. HEIGHT {
			let y_scale = y as f64 / HEIGHT as f64 * 2.0 - 1.0;

			let mut ray = Ray::new(
				&camera,
				x_scale * fov_multiplier,
				y_scale * fov_multiplier,
			);

			let mut i = 0;

			let termination = loop {
				if let Some(termination) = ray.terminate() {
					break termination;
				}

				ray.step(delta_t);

				i += 1;

				if i % 10 == 0 {
					ray.renormalize();
				}
			};

			let color = match termination {
				Termination::EventHorizon(_) => [0, 0, 0].into(),
				Termination::Background(coords) => {
					determine_pixel(coords, background)
				},
			};

			image.put_pixel(x, y, color);

			if (x * HEIGHT + y + 1) % 500 == 0 {
				let completion = (x * HEIGHT + y + 1) as f64 / (WIDTH * HEIGHT) as f64;

				let estimated_remaining = start.elapsed().div_f64(completion).mul_f64(1.0 - completion);
				dbg!(estimated_remaining);
			}
		}
	}

	println!("{:?}", start.elapsed());

	image.save("out.png").unwrap();
}

struct Camera {
	camera_position: XYZCoords,

	// I should add a check that these are orthogonal

	camera_facing: XYZUnitCoords,
	camera_right: XYZUnitCoords,
	camera_up: XYZUnitCoords,
}

#[derive(Copy, Clone, Debug)]
struct Ray {
	position: SpacetimeCoords,
	velocity: SpacetimeCoords,
	plane_x: XYZUnitCoords,
	plane_y: XYZUnitCoords,
}

impl Ray {
	fn new(camera: &Camera, x_scale: f64, y_scale: f64) -> Self {
		let Camera {
			camera_position,
			camera_facing,
			camera_right,
			camera_up,
		} = camera;

		let (unit_position, len) = camera_position.to_unit_and_len().unwrap();

		let position = SpacetimeCoords::new(
			0.0,
			len,
			0.0,
		);

		let cartesian_velocity = XYZCoords {
			x: camera_facing.x() + x_scale * camera_right.x() + y_scale * camera_up.x(),
			y: camera_facing.y() + x_scale * camera_right.y() + y_scale * camera_up.y(),
			z: camera_facing.z() + x_scale * camera_right.z() + y_scale * camera_up.z(),
		};

		let radial_velocity = XYZCoords::from(unit_position).dot(&cartesian_velocity);
		let tangent_velocity = XYZCoords {
			x: cartesian_velocity.x - radial_velocity * unit_position.x(),
			y: cartesian_velocity.y - radial_velocity * unit_position.y(),
			z: cartesian_velocity.z - radial_velocity * unit_position.z(),
		};

		// plane_x and plane_y are supposed to be basis vectors for the plane the light ray lies in
		let plane_x = unit_position;
		let plane_y;
		let theta_velocity;

		if let Some((unit, len)) = tangent_velocity.to_unit_and_len() {
			plane_y = unit;
			theta_velocity = len;
		} else {
			plane_y = plane_x; // I guess this is an ok default
			theta_velocity = 0.0;
		}

		let local_metric_inverse = metric_inverted(&position);

		let inv_local_len = (radial_velocity.powi(2) + theta_velocity.powi(2)).sqrt().recip();

		let velocity = SpacetimeCoords::new(
			-(-local_metric_inverse[[0, 0]]).sqrt(),
			local_metric_inverse[[1, 1]].sqrt() * inv_local_len * radial_velocity,
			local_metric_inverse[[2, 2]].sqrt() * inv_local_len * theta_velocity,
		);

		Ray {position, velocity, plane_x, plane_y}
	}

	fn step(&mut self, delta_t: f64) {
		let Ray {position, velocity, ..} = self;
		let mut acceleration = SpacetimeCoords::new(0.0, 0.0, 0.0);

		let local_christoffel = christoffel(position);

		for mu in 0 .. NUM_DIMS {
			let mut sum = 0.0;

			for alpha in 0 .. NUM_DIMS {
				for beta in 0 .. NUM_DIMS {
					sum += local_christoffel[[mu, alpha, beta]] * velocity[alpha] * velocity[beta];
				}
			}

			acceleration[mu] = -sum;
		}

		for i in 0 .. NUM_DIMS {
			position[i] += velocity[i] * delta_t + 0.5 * acceleration[i] * delta_t * delta_t;
		}

		for i in 0 .. NUM_DIMS {
			velocity[i] += acceleration[i] * delta_t;
		}
	}

	fn terminate(&self) -> Option<Termination> {
		let Ray {plane_x, plane_y, position, ..} = self;

		let unit_coords = || {
			let x_component = position.phi().cos();
			let y_component = position.phi().sin();

			unsafe {
				XYZUnitCoords::new_unchecked(
					x_component * plane_x.x() + y_component * plane_y.x(),
					x_component * plane_x.y() + y_component * plane_y.y(),
					x_component * plane_x.z() + y_component * plane_y.z(),
				)
			}
		};

		if *position.r() <= RS * 1.05 {
			return Some(Termination::EventHorizon(unit_coords().into()));
		}

		if *position.r() >= 20.0 {
			return Some(Termination::Background(unit_coords().into()));
		}

		None
	}

	fn renormalize(&mut self) {
		let local_metric = metric(&self.position);

		let velocity = &mut self.velocity;

		let mut space_components = 0.0;

		space_components += local_metric[[1, 1]] * velocity.r() * velocity.r();
		space_components += local_metric[[2, 2]] * velocity.phi() * velocity.phi();

		let required_time_component = -space_components;
		let time_len_squared = required_time_component / local_metric[[0, 0]];

		*velocity.t_mut() = -time_len_squared.sqrt();

		let mut cartesian_len = 0.0;

		for i in 0 .. NUM_DIMS {
			cartesian_len += velocity[i].powi(2);
		}

		let scaling = cartesian_len.sqrt().recip();

		for i in 0 .. NUM_DIMS {
			velocity[i] *= scaling;
		}
	}

	fn interval(ray: &Ray) -> f64 {
		let local_metric = metric(&ray.position);
	
		let mut sum = 0.0;
	
		for i in 0 .. NUM_DIMS {
			sum += local_metric[[i, i]] * ray.velocity[i].powi(2);
		}
	
		return sum;
	}
}

#[derive(Copy, Clone, Debug)]
enum Termination {
	EventHorizon(SphericalCoords),
	Background(SphericalCoords),
}

const C: f64 = 1.0;
const RS: f64 = 0.1;

// Schwarzschild metric
// currently only diagonal metrics are supported
fn metric(point: &SpacetimeCoords) -> Tensor<Index2> {
	let mut out = Tensor::new();

	out[[0, 0]] = -(1.0 - RS / point.r()) * C.powi(2);
	out[[1, 1]] = 1.0 / (1.0 - RS / point.r());
	out[[2, 2]] = point.r() * point.r();

	out
}

fn metric_inverted(point: &SpacetimeCoords) -> Tensor<Index2> {
	let mut out = Tensor::new();

	out[[0, 0]] = -1.0 / ((1.0 - RS / point.r()) * C.powi(2));
	out[[1, 1]] = 1.0 - RS / point.r();
	out[[2, 2]] = point.r().powi(-2);

	out
}

fn metric_partials(point: &SpacetimeCoords) -> Tensor<Index3> {
	let mut out = Tensor::new();

	out[[0, 0, 1]] = -C.powi(2) * RS / point.r().powi(2);
	out[[1, 1, 1]] = -RS / (RS - point.r()).powi(2);
	out[[2, 2, 1]] = 2.0 * point.r();

	out
}

fn christoffel(point: &SpacetimeCoords) -> Tensor<Index3> {
	let g_inv = metric_inverted(point);
	let g_partials = metric_partials(point);
	let mut gamma = Tensor::new();

	for i in 0 .. NUM_DIMS {
		let ginvii = 0.5 * g_inv[[i, i]];

		for m in 0 .. NUM_DIMS {
			let adding = ginvii * g_partials[[i, i, m]];

			gamma[[i, i, m]] += adding;
			gamma[[i, m, i]] += adding;
			gamma[[i, m, m]] -= ginvii * g_partials[[m, m, i]];
		}
	}

	gamma
}

fn determine_pixel(coords: SphericalCoords, background: &ImageBuffer::<Rgba<u8>, Vec<u8>>) -> Rgb<u8> {
	let background_width = background.width() as f64;
	let background_height = background.height() as f64;

	let SphericalCoords {mut theta, mut phi} = coords;

	theta = theta.rem_euclid(TAU);
	if theta >= PI {
		phi += PI;
		theta -= PI;
	}
	phi = phi.rem_euclid(TAU);

	let x = phi / TAU * background_width;
	let y = theta / PI * background_height;

	let pixel = background.get_pixel(x.floor() as u32, y.floor() as u32);

	[pixel[0], pixel[1], pixel[2]].into()
}
