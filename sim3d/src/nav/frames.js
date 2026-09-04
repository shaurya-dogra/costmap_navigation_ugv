// The ONE place where three.js axes meet the navigation world frame.
//
// three.js : Y up, the rover drives toward -Z at heading 0, heading increases
//            when steering LEFT (see Vehicle.jsx: driveDir = (-sin h, 0, -cos h)).
// nav world: X forward-at-heading-0, Y left, theta CCW-positive (ROS convention,
//            what perception_server / navstack expect).
//
//   nav.x = -three.z      nav.y = -three.x      nav.theta = heading
//
// Check: at heading 0 the rover moves to -Z  -> nav +X (forward).  Left of
// -Z with Y up is -X (up x forward = (0,1,0) x (0,0,-1) = (-1,0,0)) -> nav +Y.

export function toNavWorld(threeX, threeZ, heading) {
  return { x: -threeZ, y: -threeX, theta: heading };
}

export function toThree(navX, navY) {
  return { x: -navY, z: -navX };
}

// robot-frame point (x forward, y left) at a nav pose -> nav world
export function robotToNavWorld(rx, ry, pose) {
  const c = Math.cos(pose.theta), s = Math.sin(pose.theta);
  return { x: pose.x + rx * c - ry * s, y: pose.y + rx * s + ry * c };
}
