import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import unittest

from test_run import ScheduleResult, TaskSpec
from visualize import plan_from_ticks, scenario, validate


class ReleaseVisualizationTests(unittest.TestCase):
    def fixture(self):
        robots = [SimpleNamespace(robot_id=1, start_vertex=1, entry_vertex=0, release_time=2, battery=3)]
        tasks = [TaskSpec(7, 1, 2)]
        result = ScheduleResult(True, 4, {
            1: [(1, None, None, 3)], 2: [(1, 0, None, 3)],
            3: [(1, 1, None, 2)], 4: [(1, 2, 7, 1)]}, [], "FEASIBLE")
        return robots, tasks, result

    def test_scenario_preserves_true_appearance_and_metadata(self):
        robots, tasks, result = self.fixture()
        pos, hold = plan_from_ticks(robots, tasks, result)
        self.assertEqual(pos, [[None, None, 0, 1, 2]])
        view = scenario(3, [(0, 1), (1, 2)], robots, tasks, result, "entry", coords=[[0, 0], [0.5, 0], [1, 0]])
        self.assertEqual(view["delivered"], 1)
        self.assertEqual(view["release_times"], [2])
        self.assertEqual(view["entry_vertices"], [0])
        self.assertEqual(view["pos"], pos)

    def test_visual_validation_rejects_wrong_entry(self):
        robots, tasks, result = self.fixture()
        pos, hold = plan_from_ticks(robots, tasks, result)
        pos[0][2] = 1
        with self.assertRaises(AssertionError):
            validate(3, [(0, 1), (1, 2)], robots, tasks, pos, hold, 4)

    @unittest.skipUnless(shutil.which("node"), "Node is needed to execute frontend functions")
    def test_frontend_waits_off_map_until_release_and_keeps_full_battery(self):
        robots, tasks, result = self.fixture()
        view = scenario(3, [(0, 1), (1, 2)], robots, tasks, result, "entry", coords=[[0, 0], [0.5, 0], [1, 0]])
        template = Path(__file__).resolve().parents[1].joinpath("viz_ops_template.html").read_text(encoding="utf-8")
        prepare = template.split("function prepare(s) {", 1)[1].split("const pctAt", 1)[0]
        status = template.split("function rstatus(r, k) {", 1)[1].split("function orderState", 1)[0]
        screen = template.split("function slotScr(r, t) {", 1)[1].split("/* ------------------------------------------------------------------ canvas primitives", 1)[0]
        script = """
const assert = require('node:assert/strict');
let S = JSON.parse(require('node:fs').readFileSync(0, 'utf8'));
const rid = r => String(r), oid = o => String(o), plural = () => '', fmt1 = String;
const COL = { red:'red', green:'green', cyan:'cyan' };
const fit = { GV:[0, 1] }, vs = 1;
""" + "function prepare(s) {" + prepare + "function rstatus(r, k) {" + status + "function slotScr(r, t) {" + screen + """
prepare(S); S.scr = S.coords;
assert.deepEqual(S.used[0], [0,0,0,1,2]);
assert.equal(S.stuck[0][0], false);
assert.equal(rstatus(0,0)[0], 'Ожидает выхода');
assert.equal(rstatus(0,2)[0], 'В пути');
assert.equal(robotScreen(0,0)[0], null);
assert.equal(robotScreen(1,0.99)[0], null);
assert.deepEqual(robotScreen(2,0)[0], [0,0]);
const waiting = {...S, T:1, makespan:0, pos:[[null,null]], hold:[[null,null]],
  release_times:[1], orders:[], events:[]};
prepare(waiting);
assert.equal(waiting.log.some(event => event.text.includes('вышел')), false);
"""
        checked = subprocess.run([shutil.which("node"), "-e", script], input=json.dumps(view),
                                 text=True, capture_output=True, timeout=10)
        self.assertEqual(checked.returncode, 0, checked.stderr)

    def occupancy_node(self, assertions):
        template = Path(__file__).resolve().parents[1].joinpath("viz_ops_template.html").read_text(encoding="utf-8")
        prepare = template.split("function prepare(s) {", 1)[1].split("const pctAt", 1)[0]
        script = """
const assert = require('node:assert/strict');
const rid = String, oid = String, plural = () => '', fmt1 = String;
const COL = { red:'red', green:'green', cyan:'cyan' };
function scene() {
  return { T:3, makespan:3, n:3, robots:[[0,5],[2,5],[1,5]],
    edges:[[0,1,1],[1,2,1]], coords:[[0,0],[0.5,0],[1,0]], labels:null,
    orders:[[0,2]], delivered:1, events:[],
    pos:[[0,1,0,0],[2,1,2,0],[null,null,null,null]],
    hold:[[0,null,null,null],[null,0,0,null],[null,null,null,null]] };
}
""" + "function prepare(s) {" + prepare + assertions
        checked = subprocess.run([shutil.which("node"), "-e", script], text=True,
                                 capture_output=True, timeout=10)
        self.assertEqual(checked.returncode, 0, checked.stderr)

    @unittest.skipUnless(shutil.which("node"), "Node is needed to execute frontend functions")
    def test_frontend_distinguishes_physical_transfer_from_double_occupancy(self):
        self.occupancy_node("""
const actual = scene(); prepare(actual);
assert.deepEqual(actual.occupancyAt(1), [{vertex:1, robots:[0,1], kind:'transfer'}]);
assert.deepEqual(actual.transferPairs[1], [{vertex:1, robots:[0,1], cargo:0}]);
assert.equal(actual.occupancyAt(3)[0].kind, 'conflict');
const legacy = scene();
legacy.hold = legacy.hold.map(row => row.map(() => null));
legacy.events = [[1,'transfer',0,[0,1],1]];
legacy.pos[1][0] = 0;
prepare(legacy);
assert.equal(legacy.occupancyAt(0)[0].kind, 'conflict');
assert.equal(legacy.occupancyAt(1)[0].kind, 'conflict');
assert.equal(legacy.occupancyAt(3)[0].kind, 'conflict');
assert.equal(legacy.occupancyAt(1)[0].robots.length, 2);
const staleFinal = scene();
staleFinal.pos[0][3] = staleFinal.pos[1][3] = 1;
staleFinal.hold[0][3] = 0;
prepare(staleFinal);
assert.equal(staleFinal.transferPairs[3].length, 0);
assert.equal(staleFinal.occupancyAt(3)[0].kind, 'conflict');
""")

    @unittest.skipUnless(shutil.which("node"), "Node is needed to execute frontend functions")
    def test_frontend_rejects_three_robots_even_during_a_transfer(self):
        self.occupancy_node("""
const crowded = scene(); crowded.pos[2][1] = 1;
prepare(crowded);
assert.equal(crowded.transferPairs[1].length, 1);
assert.deepEqual(crowded.occupancyAt(1), [{vertex:1, robots:[0,1,2], kind:'conflict'}]);
const apart = scene(); apart.pos[0][1] = 0;
prepare(apart);
assert.equal(apart.transferPairs[1].length, 0);
assert.equal(apart.occupancyAt(1).every(slot => slot.kind === 'normal'), true);
""")


if __name__ == "__main__":
    unittest.main()
