import SwiftUI

struct MagneticStructureView: View {
    @EnvironmentObject var model: AppModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                HStack {
                    Text("Magnetic Structure").font(.title2.bold())
                    Spacer()
                    Toggle("Manual structure", isOn: $model.config.magneticStructure.enabled)
                        .toggleStyle(.switch)
                }

                if !model.config.magneticStructure.enabled {
                    ContentUnavailableView(
                        "Manual magnetic structure disabled",
                        systemImage: "wand.and.stars",
                        description: Text("The optimizer will find the classical ground state during the run. Enable the toggle to specify spin directions manually.")
                    )
                    .frame(maxWidth: .infinity)
                } else {
                    SectionCard(title: "Pattern") {
                        Picker("Pattern type", selection: $model.config.magneticStructure.patternType) {
                            Text("Antiferromagnetic").tag("antiferromagnetic")
                            Text("Generic / Custom List").tag("generic")
                        }
                        .pickerStyle(.segmented)
                        .frame(maxWidth: 380)
                    }

                    SectionCard(title: "Spin Directions (unit vectors)",
                                subtitle: "One direction per spin in the magnetic unit cell") {
                        ForEach(model.config.magneticStructure.directions.indices, id: \.self) { idx in
                            HStack {
                                Text("\(idx)")
                                    .font(.caption.monospaced())
                                    .foregroundStyle(.secondary)
                                    .frame(width: 32)
                                VectorEditor(labels: ["Sx", "Sy", "Sz"], values: directionBinding(idx))
                                Button(role: .destructive) {
                                    model.config.magneticStructure.directions.remove(at: idx)
                                } label: {
                                    Image(systemName: "trash")
                                }
                                .buttonStyle(.borderless)
                            }
                        }
                        Button {
                            model.config.magneticStructure.directions.append([1, 0, 0])
                        } label: {
                            Label("Add Direction", systemImage: "plus")
                        }
                    }
                }

                if let structure = model.magStructure {
                    SectionCard(title: "Last Minimized Structure",
                                subtitle: structure.energy.map { "Ground-state energy: \(String(format: "%.6f", $0)) meV" } ?? "") {
                        SpinStructureSceneView(structure: structure)
                            .frame(height: 380)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                        Button {
                            model.importMinimizedStructure()
                        } label: {
                            Label("Use as Manual Structure", systemImage: "square.and.arrow.down")
                        }
                    }
                }
            }
            .padding()
        }
    }

    private func directionBinding(_ idx: Int) -> Binding<[Double]> {
        Binding(
            get: {
                let dirs = model.config.magneticStructure.directions
                return idx < dirs.count ? dirs[idx] : [0, 0, 0]
            },
            set: { newVal in
                guard idx < model.config.magneticStructure.directions.count else { return }
                model.config.magneticStructure.directions[idx] = newVal
            }
        )
    }
}
