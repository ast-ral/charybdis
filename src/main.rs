mod tensor;

use std::f64::consts::PI;
use std::time::Instant;

use image::{ImageBuffer, Rgb, open};

use tensor::{Tensor, Index2, Index3};

fn main() {
	let start = Instant::now();

	const WIDTH: u32 = 1000;
	const HEIGHT: u32 = 1000;

	let mut image = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(WIDTH, HEIGHT);

	let background_dynamic = open("./background.png").unwrap();

	let background = background_dynamic.as_rgba8().unwrap();

	let background_width = background.width() as f64;
	let background_height = background.height() as f64;

	let camera_position = Coords4 {t: 0.0, r: 3.5, theta: PI / 2.0, phi: 0.0};

	let delta_t = 0.001;

	for x in 0 .. WIDTH {
		let x_scale = x as f64 / WIDTH as f64 * 2.0 - 1.0;

		for y in 0 .. HEIGHT {
			let y_scale = y as f64 / HEIGHT as f64 * 2.0 - 1.0;

			let mut ray = Ray::new(
				camera_position,
				Coords3 {r: -1.0, theta: y_scale * 0.5, phi: x_scale * 0.5},
				//Coords3 {r: x_scale, theta: -y_scale, phi: -1.0},
			);

			let mut flipped = false;

			let termination = loop {
				if let Some(termination) = ray.terminate() {
					break termination;
				}

				ray.step(delta_t);

				let theta_rem = ray.position.theta.rem_euclid(PI);

				if theta_rem < 0.15 * PI || theta_rem > 0.85 * PI {
					let Coords4 {theta: otheta, phi: ophi, ..} = ray.position;

					let (ntheta, nphi) = flip_position(otheta, ophi);

					ray.position.theta = ntheta;
					ray.position.phi = nphi;

					let Coords4 {theta: dtheta, phi: dphi, ..} = ray.velocity;

					let (dtheta, dphi) = flip_velocity(
						dtheta, dphi,
						(otheta, ophi),
						(ntheta, nphi),
					);

					ray.velocity.theta = dtheta;
					ray.velocity.phi = dphi;

					flipped = !flipped;
				}
			};

			//dbg!(termination);

			if (x * HEIGHT + y) % 100 == 0 {
				let completion = (x as f64 * HEIGHT as f64 + y as f64) / (WIDTH as f64 * HEIGHT as f64);

				let estimated_remaining = start.elapsed().div_f64(completion + 0.00000001).mul_f64(1.0 - completion);

				dbg!(estimated_remaining);
			}

			let color = match termination {
				Termination::EventHorizon(_) => [0, 0, 0].into(),
				Termination::Background(Coords2 {mut theta, mut phi}) => {
					//dbg!(theta, phi);

					if flipped {
						let (ntheta, nphi) = flip_position(theta, phi);
						theta = ntheta;
						phi = nphi;
					}

					let mut theta = theta.rem_euclid(2.0 * PI);

					if theta >= PI {
						phi += PI;
						theta -= PI;
					}

					let phi = phi.rem_euclid(2.0 * PI);

					let mut x = phi / (2.0 * PI) * background_width;
					let mut y = theta / PI * background_height;

					if x >= background_width {
						dbg!("x is {}, background_width is {}", x, background_width);
						x = background_width - 1.0;
					}

					if y >= background_height {
						dbg!("y is {}, background_height is {}", y, background_height);
						y = background_height - 1.0;
					}

					let pixel = background.get_pixel(x as u32, y as u32);

					[pixel[0], pixel[1], pixel[2]].into()
				},
			};

			image.put_pixel(x, y, color);
		}
	}

	println!("{:?}", start.elapsed());
	println!("{}ms elapsed", start.elapsed().as_millis());

	image.save("out.png").unwrap();
}

struct Ray {
	position: Coords4,
	velocity: Coords4,
}

#[derive(Copy, Clone, Debug)]
enum Termination {
	EventHorizon(Coords2),
	Background(Coords2),
}

impl Ray {
	fn new(position: Coords4, direction: Coords3) -> Self {
		let normed = direction.local_norm();
		let local_metric = metric(position);
		let velocity = Coords4 {
			t: -(-local_metric[[0, 0]]).sqrt().recip(),
			r: local_metric[[1, 1]].sqrt().recip() * normed.r,
			theta: local_metric[[2, 2]].sqrt().recip() * normed.theta,
			phi: local_metric[[3, 3]].sqrt().recip() * normed.phi,
		};

		Ray {
			position,
			velocity,
		}
	}

	fn step(&mut self, delta_t: f64) {
		let mut acceleration = [0.0; 4];

		let mut p = self.position.as_array();
		let mut v = self.velocity.as_array();
		
		let local_christoffel = christoffel(self.position);

		for mu in 0 .. 4 {
			let mut sum = 0.0;

			for alpha in 0 .. 4 {
				for beta in 0 .. 4 {
					sum += local_christoffel[[mu, alpha, beta]] * v[alpha] * v[beta];
				}
			}

			acceleration[mu] = -sum;
		}

		for i in 0 .. 4 {
			p[i] += v[i] * delta_t + 0.5 * acceleration[i] * delta_t * delta_t;
		}

		for i in 0 .. 4 {
			v[i] += acceleration[i] * delta_t;
		}

		self.position = p.into();
		self.velocity = v.into();
	}

	fn terminate(&self) -> Option<Termination> {
		if self.position.r <= RS * 1.05 {
			return Some(Termination::EventHorizon(self.position.to_coords2()));
		}

		if self.position.r >= 20.0 {
			return Some(Termination::Background(self.position.to_coords2()));
		}
		
		None
	}

	fn spacetime_interval(&self) -> f64 {
		let g = metric(self.position);
		let v = self.velocity.as_array();

		let mut sum = 0.0;

		for i in 0 .. 4 {
			for j in 0 .. 4 {
				sum += g[[i, j]] * v[i] * v[j];
			}
		}

		sum
	}
}

#[derive(Copy, Clone, Debug)]
struct Coords4 {
	t: f64,
	r: f64,
	theta: f64,
	phi: f64,
}

impl Coords4 {
	fn as_array(self) -> [f64; 4] {
		[self.t, self.r, self.theta, self.phi]
	}

	fn to_coords2(&self) -> Coords2 {
		Coords2 {
			theta: self.theta,
			phi: self.phi,
		}
	}
}

#[derive(Copy, Clone, Debug)]
struct Coords3 {
	r: f64,
	theta: f64,
	phi: f64,
}

#[derive(Copy, Clone, Debug)]
struct Coords2 {
	theta: f64,
	phi: f64,
}

impl Coords3 {
	fn local_len(&self) -> f64 {
		(self.r.powi(2) + self.theta.powi(2) + self.phi.powi(2)).sqrt()
	}

	fn local_norm(&self) -> Coords3 {
		let len = self.local_len();

		Coords3 {
			r: self.r / len,
			theta: self.theta / len,
			phi: self.phi / len,
		}
	}
}

impl From<[f64; 4]> for Coords4 {
	fn from(array: [f64; 4]) -> Coords4 {
		Coords4 {t: array[0], r: array[1], theta: array[2], phi: array[3]}
	}
}

impl From<Coords4> for [f64; 4] {
	fn from(coords: Coords4) -> [f64; 4] {
		coords.as_array()
	}
}

const C: f64 = 1.0;
const RS: f64 = 0.1;

// Schwartzschild metric
fn metric(point: Coords4) -> Tensor<Index2> {
	let mut out = Tensor::new();

	out[[0, 0]] = -(1.0 - RS / point.r) * C.powi(2);
	out[[1, 1]] = 1.0 / (1.0 - RS / point.r);
	out[[2, 2]] = point.r.powi(2);
	out[[3, 3]] = (point.r * point.theta.sin()).powi(2);

	out
}

fn metric_inverted(point: Coords4) -> Tensor<Index2> {
	let mut out = Tensor::new();

	out[[0, 0]] = -1.0 / ((1.0 - RS / point.r) * C.powi(2));
	out[[1, 1]] = 1.0 - RS / point.r;
	out[[2, 2]] = 1.0 / point.r.powi(2);
	out[[3, 3]] = 1.0 / (point.r * point.theta.sin()).powi(2);

	out
}

fn metric_partials(point: Coords4) -> Tensor<Index3> {
	let mut out = Tensor::new();

	out[[0, 0, 1]] = -C.powi(2) * RS / point.r.powi(2);
	out[[1, 1, 1]] = -RS / (RS - point.r).powi(2);
	out[[2, 2, 1]] = 2.0 * point.r;
	out[[3, 3, 1]] = 2.0 * point.r * point.theta.sin().powi(2);

	out[[3, 3, 2]] = 2.0 * point.r.powi(2) * point.theta.sin() * point.theta.cos();

	out
}

/*
// Spherical metric
fn metric(point: Coords4) -> Tensor<Index2> {
	let mut out = Tensor::new(4);

	out[[0, 0]] = -1.0;
	out[[1, 1]] = 1.0;
	out[[2, 2]] = point.r.powi(2);
	out[[3, 3]] = (point.r * point.theta.sin()).powi(2);

	out
}

fn metric_inverted(point: Coords4) -> Tensor<Index2> {
	let mut out = Tensor::new(4);

	out[[0, 0]] = -1.0;
	out[[1, 1]] = 1.0;
	out[[2, 2]] = 1.0 / point.r.powi(2);
	out[[3, 3]] = 1.0 / (point.r * point.theta.sin()).powi(2);

	out
}

fn metric_partials(point: Coords4) -> Tensor<Index3> {
	let mut out = Tensor::new(4);

	out[[2, 2, 1]] = 2.0 * point.r;
	out[[3, 3, 1]] = 2.0 * point.r * point.theta.sin().powi(2);

	out[[3, 3, 2]] = 2.0 * point.r.powi(2) * point.theta.sin() * point.theta.cos();

	out
}
*/

fn christoffel(point: Coords4) -> Tensor<Index3> {
	let g_inv = metric_inverted(point);
	let g_partials = metric_partials(point);
	let mut gamma = Tensor::new();

	for i in 0 .. 4 {
		let ginvii = 0.5 * g_inv[[i, i]];

		for m in 0 .. 4 {
			let adding = ginvii * g_partials[[i, i, m]];

			gamma[[i, i, m]] += adding;
			gamma[[i, m, i]] += adding;
			gamma[[i, m, m]] -= ginvii * g_partials[[m, m, i]];
		}
	}

	gamma
}

// these functions are both massive hacks:

fn flip_position(theta: f64, phi: f64) -> (f64, f64) {
	let x = theta.sin() * phi.cos();
	let y = theta.cos();
	let z = theta.sin() * phi.sin();

	let (x, y) = (y, x);

	let phi = z.atan2(x);
	let theta = (x * x + z * z).sqrt().atan2(y);

	(theta, phi)
}

fn flip_velocity(dtheta: f64, dphi: f64, old_position: (f64, f64), new_position: (f64, f64)) -> (f64, f64) {
	let (otheta, ophi) = old_position;
	let (ntheta, nphi) = new_position;

	let dx = otheta.cos() * ophi.cos() * dtheta - otheta.sin() * ophi.sin() * dphi;
	let dy = -otheta.sin() * dtheta;
	let dz = otheta.cos() * ophi.sin() * dtheta + otheta.sin() * ophi.cos() * dphi;

	let (dx, dy) = (dy, dx);

	let dtheta = -dy / ntheta.sin();

	let neg_sin_theta_sin_phi_dphi = dx - ntheta.cos() * nphi.cos() * dtheta;
	let sin_theta_cos_phi_dphi = dz - ntheta.cos() * nphi.sin() * dtheta;

	let sin_theta_dphi_squared = neg_sin_theta_sin_phi_dphi.powi(2) + sin_theta_cos_phi_dphi.powi(2);

	let dphi_squared = sin_theta_dphi_squared / ntheta.sin().powi(2);

	let mut dphi = dphi_squared.sqrt();

	if (ntheta.sin() * nphi.cos() * dphi).signum() != sin_theta_cos_phi_dphi.signum() {
		dphi *= -1.0;
	}

	(dtheta, dphi)
}
