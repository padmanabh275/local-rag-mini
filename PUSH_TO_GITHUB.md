# Push this project to a new GitHub repo

Follow these steps to create a new GitHub repository and push this code.

## 1. Create the repo on GitHub

1. Open [https://github.com/new](https://github.com/new).
2. Set **Repository name** to `local-rag-mini` (or any name you like).
3. Choose **Public**.
4. Do **not** add a README, .gitignore, or license (this project already has them).
5. Click **Create repository**.

## 2. Add the remote and push

In a terminal, from this project folder (`RAG`), run (replace `YOUR_USERNAME` with your GitHub username):

```bash
cd "c:\Users\Padmanabh\OneDrive\Documents\RAG"

git remote add origin https://github.com/YOUR_USERNAME/local-rag-mini.git
git branch -M main
git push -u origin main
```

If you prefer to keep the branch name `master`:

```bash
git remote add origin https://github.com/YOUR_USERNAME/local-rag-mini.git
git push -u origin master
```

If GitHub created a default `main` branch with a README, you may need to pull first and then push, or force-push (only if the remote has no other commits you need):

```bash
git remote add origin https://github.com/YOUR_USERNAME/local-rag-mini.git
git push -u origin master
# If the remote has a main branch with a README and you want to replace it:
# git push -u origin master --force
```

## 3. Update the README clone URL

In `README.md`, replace `YOUR_USERNAME` in the clone URL with your GitHub username so the clone command is correct for others.
