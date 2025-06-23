# Model Architecture: From hidden_dim to Final Action

## EPDETRVAE_EFF_IND (V6 - End Effector Independent)

```
Transformer output (hs)
   [batch, seq, hidden_dim]
            |
            +--------+--------+
            |                 |
            v                 v
    end_pose_head         eff_action_head
   [hidden_dim → 6]      [hidden_dim → 1]
            |                 |
            v                 |
       end_pose              |
      [batch, seq, 6]        |
            |                 |
            v                 |
     end_pose_mid1            |
    [Linear(6 → 512)]         |
    [ReLU activation]         |
            |                 |
            v                 |
     end_pose_mid2            |
    [Linear(512 → 256)]       |
    [ReLU activation]         |
            |                 |
            v                 v
   joints_action_head    eef_action
   [Linear(256 → action_dim-1)]  [batch, seq, 1]
            |                 |
            v                 |
     joints_action            |
   [batch, seq, action_dim-1] |
            |                 |
            +--------+--------+
                     |
                     v
                torch.cat()
                     |
                     v
                   a_hat
            [batch, seq, action_dim]
```

## EPDETRVAE (V5 - with end_pose_to_action=True)

```
Transformer output (hs)
   [batch, seq, hidden_dim]
            |
            +--------+--------+
            |                 |
            v                 v
    end_pose_head       aux_action_mid
   [hidden_dim → 6]    [Linear(hidden_dim → 256)]
            |          [ReLU activation]
            v                 |
       end_pose              |
      [batch, seq, 6]        |
            |                 |
            v                 |
     end_pose_mid1            |
    [Linear(6 → 512)]         |
    [ReLU activation]         |
            |                 |
            v                 |
     end_pose_mid2            |
    [Linear(512 → 256)]       |
    [ReLU activation]         |
            |                 |
            v                 v
       end_pose_processed   aux_mid
      [batch, seq, 256]   [batch, seq, 256]
            |                 |
            +--------+--------+
                     |
                     v
                torch.cat()
                     |
                     v
                aux_concat
            [batch, seq, 512]
                     |
                     v
               action_head
            [Linear(512 → action_dim)]
                     |
                     v
                   a_hat
            [batch, seq, action_dim]
```

## EPDETRVAE (V5 - with end_pose_to_action=False)

```
Transformer output (hs)
   [batch, seq, hidden_dim]
            |
            v
        action_head
   [Linear(hidden_dim → action_dim)]
            |
            v
          a_hat
   [batch, seq, action_dim]
```

## DETRVAE (Original - with end_pose_to_action=True)

```
Transformer output (hs)
   [batch, seq, hidden_dim]
            |
            v
    end_pose_head
   [Sequential(
     Linear(hidden_dim → 6),
     ReLU()
   )]
            |
            v
       end_pose
      [batch, seq, 6]
            |
            v
     end_pose_mid
   [Sequential(
     Linear(6 → 256),
     ReLU()
   )]
            |
            v
       end_pose_mid
      [batch, seq, 256]
            |
            v
        action_head
    [Linear(256 → action_dim)]
            |
            v
          a_hat
   [batch, seq, action_dim]
```

## DETRVAE (Original - with end_pose_to_action=False)

```
Transformer output (hs)
   [batch, seq, hidden_dim]
            |
            v
        action_head
   [Linear(hidden_dim → action_dim)]
            |
            v
          a_hat
   [batch, seq, action_dim]
```

## Key Differences

### EPDETRVAE_EFF_IND (V6):
- **End Effector Independent**: Separates gripper control from joint control
- **Two parallel paths**: 
  - End effector path: `hs → eff_action_head → eef_action (1 dim)`
  - Joint path: `hs → end_pose_head → end_pose_mid1 → end_pose_mid2 → joints_action_head → joints_action (action_dim-1)`
- **Final action**: Concatenation of `[eef_action, joints_action]`

### EPDETRVAE (V5):
- **Auxiliary concatenation**: Uses both end pose processing and direct hidden state
- **Two parallel paths that merge**:
  - End pose path: `hs → end_pose_head → end_pose_mid1 → end_pose_mid2 (256 dim)`
  - Auxiliary path: `hs → aux_action_mid (256 dim)`
- **Final action**: Concatenated features (512 dim) → `action_head` → final action

### DETRVAE (Original):
- **Simple end pose processing**: Linear transformation through pose space
- **Single path**: `hs → end_pose_head → end_pose_mid → action_head`
- **When end_pose_to_action=False**: Direct mapping `hs → action_head`

## Dimensions Summary

| Model | end_pose_to_action | Key Dimensions |
|-------|-------------------|----------------|
| EPDETRVAE_EFF_IND | True | end_pose: 6, middle_dim1: 512, middle_dim2: 256, end_eff: 1 |
| EPDETRVAE | True | end_pose: 6, middle_dim1: 512, middle_dim2: 256, aux_mid: 256 |
| EPDETRVAE | False | Direct: hidden_dim → action_dim |
| DETRVAE | True | end_pose: 6, middle_dim: 256 |
| DETRVAE | False | Direct: hidden_dim → action_dim |
