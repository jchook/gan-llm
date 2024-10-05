

# Initialize models
noise_dim = 100
embedding_dim = text_vector.shape[-1]  # Based on Llama output
generator = Generator(noise_dim, embedding_dim).to(device)
discriminator = Discriminator(embedding_dim).to(device)

# Training loop remains the same, but with embeddings
for epoch in range(epochs):
    # Train Discriminator
    optimizer_D.zero_grad()

    # Real text embeddings from Llama
    real_embeddings = text_vector  # Assume this is obtained earlier from real data
    real_labels = torch.ones(batch_size, 1).to(device)
    real_output = discriminator(real_embeddings)
    loss_real = criterion(real_output, real_labels)

    # Fake text embeddings from Generator
    noise = torch.randn(batch_size, noise_dim).to(device)
    fake_embeddings = generator(noise)
    fake_labels = torch.zeros(batch_size, 1).to(device)
    fake_output = discriminator(fake_embeddings)
    loss_fake = criterion(fake_output, fake_labels)

    # Discriminator loss
    loss_D = (loss_real + loss_fake) / 2
    loss_D.backward()
    optimizer_D.step()

    # Train Generator
    optimizer_G.zero_grad()

    # Generate fake embeddings
    noise = torch.randn(batch_size, noise_dim).to(device)
    fake_embeddings = generator(noise)

    # Generator tries to fool the Discriminator
    output = discriminator(fake_embeddings)
    loss_G = criterion(output, real_labels)

    loss_G.backward()
    optimizer_G.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Loss D: {loss_D.item()}, Loss G: {loss_G.item()}")
