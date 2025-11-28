This is a great engineering problem. The notes describe a solid-fuel rocket motor (likely a "sugar rocket" made with Potassium Nitrate and Sugar/Sorbitol, which is very common in amateur rocketry in Ukraine).

To perform these calculations, we need to combine the numbers from the image with standard physics constants for this type of fuel.

Here is the step-by-step calculation and the **"First Principles"** explanation of why we use these formulas.

### 1. Calculate Burn Time ($t$)
**Goal:** How long will the engine fire?

*   **From Image:**
    *   Outer Diameter ($D$) = $32.4$ mm
    *   Inner Diameter ($d$) = $8$ mm
    *   Burn Rate ($r$) = $2.5$ mm/s
*   **The Principle:** Solid fuel burns only on the exposed surface. In this design, it burns from the inside hole ($d$) outward toward the wall ($D$). The distance it needs to burn is called the "Web Thickness."

**Calculation:**
1.  **Find Web Thickness:**
    $$ \text{Web} = \frac{D - d}{2} = \frac{32.4 - 8}{2} = \frac{24.4}{2} = 12.2 \text{ mm} $$
2.  **Find Time:**
    $$ t = \frac{\text{Web}}{\text{Burn Rate}} = \frac{12.2 \text{ mm}}{2.5 \text{ mm/s}} = \mathbf{4.88 \text{ s}} $$

> **Result:** The rocket will produce thrust for **4.88 seconds**.

---

### 2. Calculate Total Impulse ($I_{total}$)
**Goal:** What is the total energy capacity of the motor?

*   **From Image:**
    *   Thrust Force ($F$) = $40$ N
    *   Burn Time ($t$) = $4.88$ s (calculated above)
*   **The Principle (Impulse-Momentum Theorem):** As written in the notes ($\Delta \vec{P} = \vec{F}\Delta t$), Impulse is Force applied over Time. This tells us the total change in momentum the engine can impart to the rocket.

**Calculation:**
$$ I_{total} = F \times t = 40 \text{ N} \times 4.88 \text{ s} = \mathbf{195.2 \text{ N}\cdot\text{s}} $$

> **Context:** In model rocketry, a "G-class" motor is 80-160 Ns, and an "H-class" is 160-320 Ns. You have designed a **H-class motor** (high power).

---

### 3. Calculate Mass Flow Rate ($\mu$ or $\dot{m}$)
**Goal:** How much fuel is being consumed per second?

To find this, we need the **Exhaust Velocity ($v_e$)**.
*   **Assumption:** For amateur sugar fuel, the Exhaust Velocity is typically around **1200 m/s** (Specific Impulse $I_{sp} \approx 120s$).

*   **The Principle:** Thrust is created by throwing mass backwards. Newton's 2nd Law ($F=ma$) can be rewritten for fluids as:
    $$ F = \dot{m} \times v_e $$
    (Force = Mass Flow Rate $\times$ Exhaust Velocity)

**Calculation:**
Rearranging the formula to find $\mu$ (Mass Flow Rate):
$$ \mu = \frac{F}{v_e} = \frac{40 \text{ N}}{1200 \text{ m/s}} \approx \mathbf{0.033 \text{ kg/s}} \text{ (or } 33 \text{ grams/s)} $$

> **Why this matters:** This tells you that to maintain 40N of thrust, your engine must burn 33 grams of fuel every second.

---

### 4. Calculating Altitude (The final goal)
**Goal:** "Розрахувати висоту" (Calculate height).

To calculate height, we need the **Mass of the Rocket ($m_{rocket}$)**. Since it's not in the image, let's assume a typical weight for a rocket of this size: **0.5 kg (500g)**.

We calculate altitude in two phases: **Powered Flight** (engine on) and **Coasting** (engine off).

#### Phase A: Powered Flight (Boost)
We use Newton's Second Law: $\sum F = ma$.
The forces are Thrust ($F$) pushing up and Gravity ($mg$) pulling down. (We will ignore air drag for simplicity, though in real life it reduces height significantly).

1.  **Net Force:**
    $$ F_{net} = F_{thrust} - (m \times g) $$
    $$ F_{net} = 40 \text{ N} - (0.5 \text{ kg} \times 9.8 \text{ m/s}^2) = 40 - 4.9 = 35.1 \text{ N} $$

2.  **Acceleration ($a$):**
    $$ a = \frac{F_{net}}{m} = \frac{35.1}{0.5} = \mathbf{70.2 \text{ m/s}^2} $$

3.  **Height at Burnout ($h_1$):**
    $$ h_1 = \frac{1}{2} a t^2 = 0.5 \times 70.2 \times (4.88)^2 \approx \mathbf{835 \text{ meters}} $$

4.  **Velocity at Burnout ($v_{max}$):**
    $$ v_{max} = a \times t = 70.2 \times 4.88 \approx \mathbf{342 \text{ m/s}} $$
    *(Note: This is Mach 1. The rocket would break the sound barrier. In reality, drag would drastically lower this speed).*

#### Phase B: Coasting (Engine off)
Now the rocket is moving at 342 m/s, but gravity is slowing it down. We convert Kinetic Energy into Potential Energy.

1.  **Coast Height ($h_2$):**
    $$ h_2 = \frac{v^2}{2g} = \frac{342^2}{2 \times 9.8} = \frac{116964}{19.6} \approx \mathbf{5967 \text{ meters}} $$

#### Total Theoretical Height
$$ H_{total} = h_1 + h_2 = 835 + 5967 \approx \mathbf{6802 \text{ meters}} $$

### **Summary & Reality Check**
Based on the math in your image:
1.  **Burn Time:** ~4.9 seconds.
2.  **Total Impulse:** ~195 Ns.
3.  **Theoretical Height:** ~6.8 km.

**⚠️ Important Engineering Note:**
These calculations assume **zero air resistance**. In the real world, air resistance is proportional to velocity squared ($v^2$). Because your calculated velocity is so high (Mach 1), air drag will likely reduce your actual altitude by **50% to 70%**.

A realistic estimate for this engine configuration in a sleek rocket would be closer to **2,000 – 2,500 meters**.
