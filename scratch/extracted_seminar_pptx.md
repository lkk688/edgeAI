# Extracted Content from docs/slides/autonomous/AI_Robotics_Autonomous_Systems_Seminar.pptx

## Slide 1

#### Body Content:
- seminar
- AI Robotics &
- Autonomous Systems
- What should students learn as their
- innovation foundation model?
- • Robots, self-driving cars, and drones are converging
- • Foundation models are moving from screens to the physical world
- • We need a new learning foundation for student innovation
- For high-school students and parents
- 1

---

## Slide 2

#### Body Content:
- ROADMAP
- Today’s seminar in one map
- A non-technical but research-aware tour of AI in the physical world.
- 1. The field
- How robotics, AVs and drones are changing
- 2. The model shift
- From task models to LLM, VLM, VLA and world models
- 3. The student question
- What learning foundation compounds over years?
- 4. The lab path
- Open platforms: LeRobot, Jetson, ROS 2, PX4
- “Innovation is not a single class; it is a platform that keeps paying back.”
- 2

---

## Slide 3

#### Body Content:
- BIG IDEA
- What is a “foundation model”?
- The phrase has two meanings in this talk.
- AI meaning
- A large reusable model trained on broad data, then adapted to many tasks. Examples: LLMs, VLMs, VLAs and world foundation models.
- Student innovation meaning
- A learning platform that compounds: students can keep using it for later courses, projects, internships and research.
- The central question
- In the AI age, what should become the new foundation model for student innovation?
- Definition framed for this seminar; AI foundation-model terminology reflects modern LLM/VLM/VLA usage.
- 3

---

## Slide 4

#### Body Content:
- TECHNOLOGY ARC
- The foundation model keeps changing
- Every era gives students a different “compounding skill stack.”
- Math & physics
- Coding
- Deep learning
- LLMs
- Physical AI
- Before AI
- Mathematics, physics, mechanics and problem solving were the long-term base.
- 2010s
- Programming, data structures and Python became the practical innovation base.
- 2016–2022
- Deep learning and perception models changed robotics and driving.
- 2023–now
- LLMs and multimodal models added reasoning, language and agents.
- Next
- VLA + world models + real labs.
- Takeaway: students should learn platforms that evolve with the research frontier—not tools that expire when a vendor changes direction.
- 4

---

## Slide 5

#### Body Content:
- WHY HARD
- Why autonomous systems are different from phone apps
- They must reason and act in the real world, where mistakes have physical consequences.
- They must perceive
- What is around me? Objects, roads, people, obstacles, weather.
- They must predict
- What will others do next? What might happen if I act?
- They must plan
- Which action is safe, legal, efficient and comfortable?
- They must be verified
- Not only “works in demos,” but fails safely in rare cases.
- Core autonomy loop: sensing → prediction → planning → control → safety validation.
- 5

---

## Slide 6

#### Body Content:
- SYSTEMS
- One AI wave, many autonomous systems
- Cars, robots and drones now share more of the same AI stack than ever before.
- Self-driving cars
- Robotic arms
- Humanoids
- Drones
- Delivery robots
- Common ingredients: sensors + compute + simulation + control + learning + safety.
- 6

---

## Slide 7

#### Body Content:
- HISTORY
- Phase 1: Robots were mostly programmed
- Classical robotics built the first foundation: geometry, control and estimation.
- Maps & localization
- Where am I? Use GPS, IMU, LiDAR, SLAM and HD maps.
- Rules & planning
- Path planners and behavior trees encode expert knowledge.
- Control
- PID, MPC and vehicle dynamics turn plans into motion.
- “Classical autonomy was powerful, but brittle: every new edge case needed new engineering.”
- This is why modern systems still need math and physics—but they also need data-driven learning.
- 7

---

## Slide 8

#### Body Content:
- DEEP LEARNING
- Phase 2: Deep learning made perception practical
- CNNs changed how machines recognized objects, lanes and road scenes.
- Object detection
- R-CNN → Faster R-CNN → YOLO / SSD made real-time detection practical.
- Segmentation
- FCN, SegNet, ENet and related models learned road/lane pixels.
- LiDAR learning
- PointNet, VoxelNet, SECOND and PointPillars learned sparse 3D data.
- But systems still looked like a “bag of models”
- one model for cars, one for lanes, one for traffic lights, one for depth…
- Representative works: Faster R-CNN, YOLO, SSD, PointNet, VoxelNet, SECOND, PointPillars.
- 8

---

## Slide 9

#### Body Content:
- DRIVING EXAMPLE
- Autonomous driving became a systems race
- The winners were not just better models—they built data, simulation, hardware and operations.
- Sensors
- cameras / LiDAR / radar / GPS / IMU
- Data
- fleets, maps, labels, edge cases
- Compute
- on-car AI chips and cloud training
- Safety
- simulation, testing, monitoring
- Ops
- robotaxi, trucking, ADAS, delivery
- Robotaxi path
- Waymo, Cruise, Baidu Apollo, Pony.ai, Zoox: expensive L4 stacks and careful city-by-city deployment.
- Consumer ADAS path
- Tesla, Mobileye, GM Super Cruise, Chinese EV makers: high-volume L2/L2+ systems.
- Truck / logistics path
- Aurora, Waabi and others focus on narrower highway/logistics domains.
- Insight for students: autonomy is interdisciplinary—AI plus hardware plus software plus validation.
- 9

---

## Slide 10

#### Body Content:
- BEV
- Why everyone moved toward Bird’s-Eye View
- Driving is a 3D spatial problem; BEV gives the model a map-like workspace.
- Before BEV
- Camera models saw each image separately; fusion happened late and inconsistently.
- BEV representation
- Convert multi-camera and LiDAR features into a shared top-down coordinate system.
- Why it matters
- Planning, prediction, occupancy, maps and collision checking all become easier.
- Representative works: Lift-Splat-Shoot, BEVDet, BEVFormer, BEVFusion, occupancy networks.
- 10

---

## Slide 11

#### Body Content:
- TESLA CASE
- Tesla as a case study: from HydraNet to End-to-End
- A simplified view of the architectural shift.
- HydraNet
- shared backbone + many task heads
- Vector Space / BEV
- multi-camera fusion into a unified spatial scene
- Occupancy
- dense world representation beyond boxes and lanes
- End-to-End
- network learns more of the perception → planning stack
- Why this mattered
- • Multi-task learning reduced duplicated perception networks
- • BEV / occupancy reduced hand-engineered fusion and improved spatial consistency
- • End-to-end training turned “manual interfaces” into learned interfaces
- Publicly discussed in Tesla AI Day materials; simplified here for education, not as a system specification.
- 11

---

## Slide 12

#### Body Content:
- END-TO-END
- End-to-End is not magic—it changes where the interfaces live
- The goal is fewer brittle hand-coded boundaries, but verification becomes harder.
- Modular stack
- Perception → prediction → planning → control. Easy to debug, but interfaces can lose information.
- End-to-End stack
- Train larger parts jointly, sometimes from sensors and route to trajectory/action.
- Research problem
- How do we get scalability without losing interpretability, safety and testability?
- Examples: Tesla FSD v12 discussions, Waymo EMMA, Wayve end-to-end autonomy research.
- 12

---

## Slide 13

#### Body Content:
- WORLD MODELS
- World models are back—but now at foundation-model scale
- A world model learns how the environment evolves, so agents can imagine outcomes before acting.
- Old idea
- Learning an internal simulator has been a dream in AI and robotics for decades.
- Why now?
- Video transformers, diffusion models, large-scale data and GPUs make generative prediction practical.
- Why it matters
- Synthetic data, rare-event generation, counterfactual testing and planning in imagination.
- Examples: Wayve GAIA-1, Waabi neural simulator, NVIDIA Cosmos world foundation models.
- Sources: Wayve GAIA-1 technical report; NVIDIA Cosmos World Foundation Model Platform; Waabi website.
- 13

---

## Slide 14

#### Body Content:
- LLM IMPACT
- Why LLMs changed robotics and autonomous driving
- LLMs did not solve driving by themselves—but they changed the architecture of intelligence.
- Language became an interface
- Humans can specify tasks, constraints, goals and explanations in natural language.
- Reasoning became programmable
- Planning, tool use, code generation and self-checking became part of AI workflows.
- Multimodal models emerged
- Images, video, text, maps, actions and sensor data can be tokenized together.
- “The important shift: robots can now combine perception, language, memory, tools and action in one learning loop.”
- • VLM: vision + language understanding
- • VLA: vision + language + action output
- • Agentic AI: plans, calls tools, checks results, improves workflows
- 14

---

## Slide 15

#### Body Content:
- VLA
- VLA: when a model outputs actions, not just answers
- Vision-Language-Action models are becoming a bridge between AI and physical robots.
- RT-2
- DeepMind: converts VLM knowledge into robot actions using web + robot data.
- OpenVLA
- Open-source 7B VLA trained on large robot demonstrations; adaptable to new robots.
- π0 / π0-FAST
- Physical Intelligence: VLA policies for general robot control with action experts / tokenization.
- VLA changes the educational story: a simple robot arm can now connect to the same research trend as frontier robot learning.
- Sources: DeepMind RT-2; OpenVLA project and CoRL paper; Physical Intelligence π0 paper.
- 15

---

## Slide 16

#### Body Content:
- WHO IS BUILDING
- Startups are racing toward “physical intelligence”
- The frontier is moving from models that chat to models that act.
- Wayve
- world models + end-to-end driving
- Waabi
- neural simulator + autonomous trucks
- Physical Intelligence
- π0 VLA robot policy
- Skild AI
- robotics foundation model
- Covariant
- RFM-1 for warehouse robots
- Figure / 1X
- humanoid robots with AI models
- “The startup pattern: build data engines, train generalist models, then specialize to cars, trucks, warehouses, homes or factories.”
- Sources: Wayve GAIA-1, Waabi website, Physical Intelligence π0, Skild AI, Covariant RFM-1, Figure/1X websites.
- 16

---

## Slide 17

#### Body Content:
- DRONES
- Drones show why open platforms matter
- A drone is a compact autonomous system: sensing, estimation, control, safety and mission planning.
- PX4
- Open-source autopilot for drones and other unmanned vehicles.
- MAVLink
- Communication protocol connecting flight controller, ground station and companion computer.
- ROS 2 / simulation
- Bridge perception, planning and autonomy with reproducible experiments.
- Good drone education should go beyond “press a button to fly.” Students should learn the stack they can modify.
- • State estimation: IMU, GPS, visual odometry
- • Planning: waypoints, obstacle avoidance, path optimization
- • Safety: geofencing, failsafe behavior, flight logs
- Source: PX4 official site and PX4-Autopilot GitHub.
- 17

---

## Slide 18

#### Body Content:
- STUDENT FOUNDATION
- What is a student innovation foundation model?
- Our definition: a learning platform that keeps enabling future learning and invention.
- Transferable
- Useful across robotics, drones, AVs and AI systems
- Open
- Students can inspect, modify and extend it
- Composable
- Connects sensors, models, simulation, hardware and code
- Research-connected
- Aligned with what universities and labs actually use
- Affordable path
- Can start simple, then grow toward serious projects
- Portfolio-ready
- Produces demos, data, reports and code students can show
- “A foundation course is not a toy demo. It is a launchpad.”
- 18

---

## Slide 19

#### Body Content:
- WARNING
- Not every robotics course becomes a foundation
- Some courses are exciting but do not compound into long-term innovation.
- Vendor-specific learning
- Easy demos, proprietary apps, closed APIs, limited access to algorithms. Good for motivation, weak for research continuity.
- Foundation learning
- Open software, programmable hardware, reusable datasets, simulation and models students can modify.
- The test
- Can a student use the same platform later for a science fair, college research, internship project or startup prototype?
- 19

---

## Slide 20

#### Body Content:
- OPEN STACK
- A practical foundation stack for AI-era students
- Start simple, but align with research-grade ecosystems.
- Math & physics
- geometry, probability, dynamics
- Coding
- Python, Git, Linux, data
- AI models
- PyTorch, transformers, VLM/VLA
- Robotics middleware
- ROS 2, sensors, simulation
- Embedded AI
- NVIDIA Jetson, CUDA/TensorRT
- Open autonomy
- LeRobot, PX4, datasets, benchmarks
- This is the modern equivalent of “math + coding,” expanded for physical AI.
- 20

---

## Slide 21

#### Body Content:
- ROBOT ARM PATH
- Robot pathway: start with a simple arm, connect to frontier VLA
- Small hardware can teach the same loop used in serious robot-learning research.
- Step 1: Teleoperation
- Collect demonstrations by manually guiding a simple arm.
- Step 2: Imitation learning
- Train policies on student-collected datasets.
- Step 3: LeRobot / SO-ARM101
- Use open-source tooling aligned with modern robot-learning workflows.
- Why this matters: students learn data collection, control, model training and evaluation—not just robot assembly.
- Sources: Hugging Face LeRobot documentation; SO-101 documentation; OpenVLA / π0 robot-learning papers.
- 21

---

## Slide 22

#### Body Content:
- JETSON PATH
- Embedded AI pathway: why Jetson is useful
- Students learn the gap between “model works on laptop” and “model works on a robot.”
- Edge inference
- Run vision models locally with latency, memory and power constraints.
- Robot integration
- Connect cameras, sensors, ROS 2 nodes and control loops.
- Research continuity
- Same NVIDIA ecosystem connects Jetson, Isaac ROS, Isaac Sim, CUDA, TensorRT and robotics models.
- Representative ecosystem: NVIDIA Jetson, Isaac ROS, Isaac Sim, CUDA/TensorRT.
- 22

---

## Slide 23

#### Body Content:
- DRONE PATH
- Drone pathway: PX4 as a foundation platform
- A serious drone pathway should connect flight control to AI autonomy.
- Flight control
- PX4, ArduPilot, control loops, failsafes and logs.
- Companion computer
- Jetson / Raspberry Pi runs perception, planning and communication.
- Autonomy stack
- ROS 2, MAVLink, simulation, perception and mission planning.
- “PX4 is valuable because students can go beyond flying—into modifying, logging, simulating and researching autonomy.”
- • Beginner: waypoint mission + flight logs
- • Intermediate: vision-based landing or tracking
- • Advanced: multi-agent drones, SLAM, reinforcement learning or safety verification
- Sources: PX4 official site; PX4-Autopilot GitHub; MAVLink ecosystem.
- 23

---

## Slide 24

#### Body Content:
- AUTONOMOUS DRIVING PATH
- Autonomous-driving pathway: research-grade data without a car
- Students can study serious autonomy using public datasets, simulation and open models.
- Datasets
- KITTI, nuScenes, Waymo, Argoverse, BDD100K.
- Tasks
- 2D/3D detection, tracking, BEV segmentation, occupancy, prediction.
- Research skills
- Evaluation metrics, failure analysis, ablations, latency and deployment tradeoffs.
- A high-school project can be more than “detect cars”: it can ask a research question.
- 24

---

## Slide 25

#### Body Content:
- AGENTIC AI
- Agentic AI: the new research assistant for builders
- Students can use AI agents to accelerate—but not replace—engineering thinking.
- Code assistant
- Generate ROS 2 nodes, scripts, tests and documentation—then verify.
- Data analyst
- Summarize logs, plot metrics, find failure cases, compare experiments.
- Experiment planner
- Suggest ablations, safety checks and research hypotheses.
- Best practice: use AI to expand iteration speed, but keep humans responsible for correctness, safety and originality.
- 25

---

## Slide 26

#### Body Content:
- PROJECT IDEAS
- What can students build in 3–6 months?
- Projects should create artifacts: code, data, demo video, analysis and reflection.
- Robot arm imitation learning
- Collect 100 demonstrations; train a policy; analyze failures.
- Drone visual landing
- Use PX4 simulation + camera model to land on a marker.
- Jetson real-time perception
- Deploy YOLO / depth / tracking with latency and power measurements.
- BEV mini-project
- Use public driving data to compare camera vs LiDAR or 2D vs BEV.
- World-model demo
- Use video generation/simulation to create rare scenarios and discuss limits.
- Agentic robot logger
- Build an AI assistant that reads robot logs and suggests debugging steps.
- 26

---

## Slide 27

#### Body Content:
- RESEARCH LADDER
- A research-style learning ladder
- The goal is not to memorize buzzwords; it is to learn how to ask better questions.
- Level 0
- Run an existing demo and explain each component
- Level 1
- Change one variable and measure the effect
- Level 2
- Build a small pipeline and evaluate failures
- Level 3
- Pose an original research question and test it
- “Research thinking begins when a student asks: What changed? Why did it fail? How do I know?”
- 27

---

## Slide 28

#### Body Content:
- FOR PARENTS
- What should parents look for?
- A strong program should produce durable skill, not just a flashy demo.
- Evidence of learning
- Can the student explain how the system works and why it fails?
- Open-ended building
- Can the student modify code, collect data and test alternatives?
- Research continuity
- Can the project grow into a portfolio, science fair, paper or internship?
- • Avoid: closed toys, one-click demos, vendor-only curricula
- • Prefer: open platforms, reproducible experiments, real code, logs, datasets
- • Reward: curiosity, persistence, careful analysis and teamwork
- The best outcome is not “my child used AI.” It is “my child learned how to build, test and improve intelligent systems.”
- 28

---

## Slide 29

#### Body Content:
- TAKEAWAY
- The new foundation model for student innovation
- A proposed answer for the AI + robotics era.
- Math + Coding + AI + Open Robotics Platforms + Research Mindset
- • Math still matters because physical systems obey geometry and dynamics
- • Coding still matters because students must build and debug systems
- • AI matters because foundation models are becoming the new interface
- • Open platforms matter because they compound into future research
- “Start simple. Choose platforms that scale.”
- 29

---

## Slide 30

#### Body Content:
- REFERENCES
- Selected references and platforms
- For students who want to explore further.
- • DeepMind RT-2: vision-language-action model for robot control (2023).
- • OpenVLA: open-source VLA model; pretrained on large robot demonstrations (2024/2025).
- • Physical Intelligence π0: VLA flow model for general robot control (2024).
- • Hugging Face LeRobot and SO-101: open robot-learning tooling and low-cost arm path.
- • PX4 Autopilot: open-source flight control software for drones and unmanned vehicles.
- • Wayve GAIA-1: generative world model for autonomous driving (2023).
- • NVIDIA Cosmos: world foundation model platform for Physical AI (2025+).
- • Waabi: end-to-end interpretable model and neural simulator for autonomous trucks.
- • BEV/Driving: Lift-Splat-Shoot, BEVFormer, BEVFusion, occupancy networks.
- • Startups: Skild AI, Covariant RFM-1, Figure, 1X, Sanctuary AI and others.
- Questions?
- 30

---

