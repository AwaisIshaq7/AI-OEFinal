# GitHub Setup Guide

## Manual Setup Steps for GitHub Repository

### Step 1: Initialize Local Git Repository

Open a terminal/command prompt in the project directory and run:

```bash
cd d:\AIOEL
git init
git config user.name "Your Name"
git config user.email "your.email@example.com"
git add .
git commit -m "Initial commit: Agricultural Intelligence System"
```

### Step 2: Create GitHub Repository

1. Go to https://github.com/new
2. Enter repository name: `agricultural-intelligence-system`
3. Select: **Public** (as required by lab specifications)
4. Add description: "Multi-model agricultural decision support system integrating Decision Tree, KNN, and Linear Regression"
5. Choose License: **MIT License**
6. Click "Create repository"

### Step 3: Connect Local Repository to GitHub

After creating the repository, GitHub will show instructions. Follow these:

```bash
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/agricultural-intelligence-system.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your actual GitHub username.

### Step 4: Verify Repository Contents

After pushing, verify on GitHub that the following structure is visible:

```
├── README.md                    ✓
├── LICENSE (MIT)                ✓
├── TECHNICAL_REPORT.md          ✓
├── requirements.txt             ✓
├── .gitignore                   ✓
├── data/
│   ├── crop_recommendation.csv  ✓
│   └── DATA_DICTIONARY.md       ✓
├── src/
│   ├── download_data.py         ✓
│   ├── preprocessing.py         ✓
│   ├── models.py                ✓
│   ├── utils.py                 ✓
│   ├── train.py                 ✓
│   └── gui.py                   ✓
├── models/
│   ├── decision_tree_classifier.pkl
│   ├── knn_clustering.pkl
│   └── linear_regression.pkl
└── results/
    ├── model_results.json
    └── model_visualizations.png
```

### Step 5: Repository Visibility Settings

1. Go to **Settings** tab on GitHub
2. Scroll to **Visibility** section
3. Confirm it's set to **Public**
4. Scroll down to **Access** section
5. Ensure no branch protection rules block access

### Step 6: Add Repository Link to Submission

Copy your repository URL (format: `https://github.com/YOUR_USERNAME/agricultural-intelligence-system`)

Submit this URL via your Learning Management System (LMS).

---

## Testing the Installation

After cloning or pushing, verify the repository works correctly:

```bash
# Clone the repository (on a different machine or directory)
git clone https://github.com/YOUR_USERNAME/agricultural-intelligence-system.git
cd agricultural-intelligence-system

# Install dependencies
pip install -r requirements.txt

# Run tests
python src/train.py           # Retrain models
python src/gui.py             # Launch GUI
```

---

## Troubleshooting Common Issues

### Issue: "fatal: not a git repository"
**Solution**: Run `git init` in the project directory first

### Issue: Authentication failed when pushing
**Solution**: 
- Use SSH keys (recommended): Generate SSH key and add to GitHub account
- Or use Personal Access Token (PAT) instead of password
- GitHub disabled password authentication in 2021

### Issue: Large files cannot be pushed
**Solution**: Git has a 100 MB file size limit. For models > 100 MB:
- Use Git LFS (Large File Storage)
- Or exclude models/ from git and regenerate via `train.py`

### Issue: Repository not showing as public
**Solution**: Go to Settings → scroll to Visibility section → select "Public"

---

## Maintaining Repository During Lab

### Regular Commits

Make commits for each major milestone:

```bash
git add src/models.py
git commit -m "feat: Implement Decision Tree classifier"
git push origin main

git add src/gui.py
git commit -m "feat: Build Tkinter GUI interface"
git push origin main
```

### Commit Message Guidelines

Use conventional commits format:
- `feat: Add new feature`
- `fix: Fix bug or issue`
- `docs: Update documentation`
- `refactor: Restructure code`
- `test: Add tests`

### Final Submission Checklist

Before final submission, verify:

- [ ] Repository is public and accessible
- [ ] All files present and up-to-date
- [ ] README.md renders correctly on GitHub
- [ ] Models and data included or regeneration instructions provided
- [ ] requirements.txt lists all dependencies
- [ ] .gitignore configured to exclude unnecessary files
- [ ] No API keys or credentials committed
- [ ] Repository URL works and accessible to evaluators
- [ ] At least 3-5 meaningful commits in history

---

## File Size Considerations

### Current Estimated Sizes

| Item | Size |
|------|------|
| Code (src/) | ~25 KB |
| Documentation | ~50 KB |
| Dataset CSV | ~50 KB |
| Models (3 × .pkl) | ~100 KB |
| Results | ~150 KB |
| **Total** | **~375 KB** |

✓ Well within GitHub's repository limits

---

## GitHub Features to Enable

### 1. Enable Discussions (Optional)
- **Settings** → **General** → Check "Discussions"
- Useful for project documentation and Q&A

### 2. Add Topics
- On main repository page, under **About**
- Add: `agriculture`, `machine-learning`, `decision-tree`, `clustering`, `regression`
- Helps others discover your project

### 3. Create Releases (After Lab Completion)
- Go to **Releases** → **Create a new release**
- Tag: v1.0
- Title: "Agricultural Intelligence System v1.0"
- Attach TECHNICAL_REPORT.md as PDF

### 4. Add README Badges
```markdown
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
```

---

## After Lab Submission

### Keeping Repository Accessible

Per lab requirements, maintain repository as public for **minimum one academic semester**.

Recommended actions:
- Set reminder to keep repository public
- Monitor for any GitHub notices about account changes
- Maintain repository backup on personal storage

### Future Updates

```bash
# When adding future features
git checkout -b feature/new-feature
# Make changes
git add .
git commit -m "feat: Describe changes"
git push origin feature/new-feature

# Create Pull Request on GitHub
# Merge to main after review
```

---

## Important Notes

1. **Do NOT commit**:
   - Personal credentials or API keys
   - Large binary files (> 100 MB)
   - Virtual environment folders
   - IDE configuration files

2. **Do commit**:
   - All source code (.py files)
   - Configuration files (requirements.txt)
   - Documentation (README, reports)
   - Data files (< 100 MB)
   - Model artifacts (if < 100 MB total)

3. **Public Repository Implications**:
   - Code is visible to anyone
   - Others can fork and use your project
   - Great for portfolio; be professional
   - License (MIT) allows free use with attribution

---

**Last Updated**: May 20, 2026  
**Reference**: GitHub Official Documentation - https://docs.github.com/
